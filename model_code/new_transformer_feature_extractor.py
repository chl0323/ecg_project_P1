import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class PositionalEncoding:
    def __init__(self, position, d_model):
        self.position = position
        self.d_model = d_model
        
    def __call__(self):
        # 创建可训练的位置编码
        pos_encoding = layers.Embedding(self.position, self.d_model)(tf.range(self.position, dtype=tf.int32))
        pos_encoding = tf.expand_dims(pos_encoding, 0)  # 添加batch维度
        return pos_encoding

class TransformerFeatureExtractor:
    def __init__(self, input_dim, sequence_length, num_heads=4, d_model=128, rate=0.1):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.d_model = d_model
        self.rate = rate
        self.attention_layers = []  # 保存每层attention层对象
        
        # 初始化可训练的位置编码
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length,
            output_dim=input_dim,
            name='position_embedding'
        )

    def transformer_encoder_layer(self, x, num_heads, d_model, rate):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.input_dim // num_heads
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed forward network
        ffn_output = self.point_wise_feed_forward_network(self.input_dim, d_model)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

    def build(self):
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.input_dim))
        
        # 创建位置索引
        position_indices = tf.keras.layers.Lambda(
            lambda x: tf.range(start=0, limit=self.sequence_length, delta=1),
            output_shape=(self.sequence_length,)
        )(inputs)
        
        # 扩展维度以匹配batch size
        position_indices = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x[1], axis=0),
            output_shape=(1, self.sequence_length)
        )([inputs, position_indices])
        
        # 复制到所有batch
        position_indices = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[1], [tf.shape(x[0])[0], 1]),
            output_shape=(None, self.sequence_length)
        )([inputs, position_indices])
        
        # 添加位置编码
        pos_encoding = self.position_embedding(position_indices)
        x = tf.keras.layers.Add()([inputs, pos_encoding])
        
        # Transformer编码器层
        self.attention_layers = []
        for i in range(2):  # 使用2层transformer编码器
            # 多头自注意力层
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.input_dim // self.num_heads,
                name=f'multi_head_attention_{i}'
            )
            attn = attn_layer(x, x, return_attention_scores=False)
            self.attention_layers.append(attn_layer)
            
            # 残差连接和层归一化
            x = tf.keras.layers.LayerNormalization(name=f'layer_norm_1_{i}')(tf.keras.layers.Add()([x, attn]))
            
            # 前馈网络
            ffn = tf.keras.layers.Dense(self.d_model, activation='relu', name=f'dense_1_{i}')(x)
            ffn = tf.keras.layers.Dense(self.input_dim, name=f'dense_2_{i}')(ffn)
            x = tf.keras.layers.LayerNormalization(name=f'layer_norm_2_{i}')(tf.keras.layers.Add()([x, ffn]))
            x = tf.keras.layers.Dropout(self.rate)(x)
        
        # 确保x的维度正确
        x = tf.keras.layers.Reshape((self.sequence_length, self.input_dim))(x)
        
        # 双通道聚合策略
        # 1. Mean Pooling
        mean_pool = tf.keras.layers.GlobalAveragePooling1D(name='mean_pool')(x)
        # 2. Max Pooling
        max_pool = tf.keras.layers.GlobalMaxPooling1D(name='max_pool')(x)
        
        # 合并两种池化结果
        combined = tf.keras.layers.Concatenate(name='concat')([mean_pool, max_pool])
        
        # 最终的特征向量
        embedding = tf.keras.layers.Dense(self.input_dim, activation='tanh', name='final_dense')(combined)
        
        return tf.keras.Model(inputs, embedding)

    def get_config(self):
        """返回配置，使类可序列化"""
        return {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'rate': self.rate
        }

    @classmethod
    def from_config(cls, config):
        """从配置创建实例"""
        return cls(**config)

    def load_weights(self, weights_path):
        """
        加载模型权重，处理可能的权重不匹配情况
        """
        try:
            # 首先尝试直接加载权重
            self.model.load_weights(weights_path)
            print(f"成功加载权重文件: {weights_path}")
        except Exception as e:
            print(f"直接加载权重失败: {str(e)}")
            print("尝试使用兼容模式加载权重...")
            
            try:
                # 尝试加载完整模型
                saved_model = tf.keras.models.load_model(weights_path, compile=False)
                saved_weights_dict = {layer.name: layer.get_weights() for layer in saved_model.layers}
            except Exception as e:
                print(f"加载完整模型失败: {str(e)}")
                print("尝试加载权重文件...")
                
                # 如果文件是.weights.h5格式，直接使用load_weights
                if weights_path.endswith('.weights.h5'):
                    try:
                        self.model.load_weights(weights_path)
                        print(f"成功加载权重文件: {weights_path}")
                        return
                    except Exception as e:
                        print(f"加载权重文件失败: {str(e)}")
                        raise
            
            # 获取当前模型的层
            current_layers = {layer.name: layer for layer in self.model.layers}
            
            # 尝试加载兼容的权重
            loaded_count = 0
            for layer_name, weights in saved_weights_dict.items():
                if layer_name in current_layers:
                    try:
                        current_layers[layer_name].set_weights(weights)
                        loaded_count += 1
                        print(f"成功加载层 {layer_name} 的权重")
                    except Exception as e:
                        print(f"无法加载层 {layer_name} 的权重: {str(e)}")
            
            print(f"权重加载完成，成功加载了 {loaded_count} 个层的权重")
            
            if loaded_count == 0:
                raise ValueError("没有成功加载任何层的权重")

    def get_attention_scores(self, x_input, layer_idx=0):
        """
        获取指定层的attention scores。
        x_input: shape (batch, seq_len, input_dim)
        layer_idx: 指定第几层的attention（从0开始）
        返回: (batch, num_heads, seq_len, seq_len)
        """
        if not isinstance(x_input, tf.Tensor):
            x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
        
        # 确保输入数据形状正确
        if len(x_input.shape) != 3:
            raise ValueError(f"输入数据应该是3维的 (batch_size, sequence_length, features), 但得到的是 {x_input.shape}")
        
        if x_input.shape[1] != self.sequence_length:
            raise ValueError(f"序列长度应该是 {self.sequence_length}, 但得到的是 {x_input.shape[1]}")
        
        if x_input.shape[2] != self.input_dim:
            raise ValueError(f"特征维度应该是 {self.input_dim}, 但得到的是 {x_input.shape[2]}")
        
        # 创建位置编码
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [tf.shape(x_input)[0], 1])
        pos_encoding = self.position_embedding(positions)
        x_input = x_input + pos_encoding
        
        # 获取指定层的attention scores
        attn_layer = self.attention_layers[layer_idx]
        _, attn_scores = attn_layer(x_input, x_input, return_attention_scores=True)
        
        # 确保返回numpy数组
        return attn_scores.numpy()