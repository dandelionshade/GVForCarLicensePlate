# -*- coding: utf-8 -*-
"""
CRNN模型实现 - 车牌识别系统
基于卷积神经网络(CNN) + 循环神经网络(RNN) + CTC的端到端车牌识别模型
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

# 条件导入TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Input, Conv2D, MaxPooling2D, BatchNormalization, Activation,
        Reshape, Dense, LSTM, Bidirectional, Lambda, TimeDistributed
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow.keras.backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow不可用，CRNN模型功能将被禁用")

import cv2
from ..core.config import Config


class CRNNModel:
    """CRNN车牌识别模型"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.model = None
        self.character_set = self.config.get_character_set()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.character_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.character_set)}
        self.num_classes = len(self.character_set) + 1  # +1 for CTC blank
        
        if TF_AVAILABLE:
            self._build_model()
        else:
            self.logger.error("TensorFlow不可用，无法创建CRNN模型")

    def _build_model(self):
        """构建CRNN模型"""
        try:
            input_shape = (*self.config.MODEL_INPUT_SIZE[::-1], self.config.MODEL_CHANNELS)  # (height, width, channels)
            
            # 输入层
            inputs = Input(shape=input_shape, name='input_image')
            
            # CNN特征提取部分
            # Block 1
            x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
            x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
            x = MaxPooling2D((2, 2), name='pool1')(x)
            x = BatchNormalization(name='bn1')(x)
            
            # Block 2
            x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
            x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
            x = MaxPooling2D((2, 2), name='pool2')(x)
            x = BatchNormalization(name='bn2')(x)
            
            # Block 3
            x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
            x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2')(x)
            x = MaxPooling2D((2, 1), name='pool3')(x)  # 只在高度方向池化
            x = BatchNormalization(name='bn3')(x)
            
            # Block 4
            x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(x)
            x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2')(x)
            x = MaxPooling2D((2, 1), name='pool4')(x)  # 只在高度方向池化
            x = BatchNormalization(name='bn4')(x)
            
            # 特征图转换为序列
            # 假设经过卷积和池化后，特征图尺寸为 (batch, 2, width/4, 256)
            new_shape = (tf.shape(x)[2], tf.shape(x)[1] * tf.shape(x)[3])  # (width/4, height*channels)
            x = Reshape(target_shape=new_shape, name='reshape')(x)
            
            # 全连接层降维
            x = TimeDistributed(Dense(256, activation='relu'), name='dense1')(x)
            x = TimeDistributed(Dense(256, activation='relu'), name='dense2')(x)
            
            # RNN序列建模部分
            # 双向LSTM
            x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), name='bilstm1')(x)
            x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), name='bilstm2')(x)
            
            # 输出层
            outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
            
            # 创建模型
            self.model = Model(inputs=inputs, outputs=outputs, name='CRNN_PlateRecognition')
            
            # 编译模型
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=self._ctc_loss,
                metrics=['accuracy']
            )
            
            self.logger.info("CRNN模型构建成功")
            self.logger.info(f"模型参数量: {self.model.count_params():,}")
            
        except Exception as e:
            self.logger.error(f"CRNN模型构建失败: {e}")
            self.model = None

    def _ctc_loss(self, y_true, y_pred):
        """CTC损失函数"""
        if not TF_AVAILABLE:
            return None
            
        # 获取序列长度
        input_length = tf.cast(tf.shape(y_pred)[1], tf.int32)
        label_length = tf.cast(tf.shape(y_true)[1], tf.int32)
        
        input_length = input_length * tf.ones(shape=(tf.shape(y_pred)[0], 1), dtype=tf.int32)
        label_length = label_length * tf.ones(shape=(tf.shape(y_true)[0], 1), dtype=tf.int32)
        
        # 计算CTC损失
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def load_model(self, model_path: str = None) -> bool:
        """加载预训练模型"""
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow不可用")
            return False
            
        if model_path is None:
            model_path = self.config.CRNN_MODEL_PATH
        
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'_ctc_loss': self._ctc_loss}
                )
                self.logger.info(f"模型加载成功: {model_path}")
                return True
            else:
                self.logger.warning(f"模型文件不存在: {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False

    def save_model(self, model_path: str = None):
        """保存模型"""
        if not TF_AVAILABLE or self.model is None:
            self.logger.error("模型不可用")
            return False
            
        if model_path is None:
            model_path = self.config.CRNN_MODEL_PATH
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            self.logger.info(f"模型保存成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """预测车牌号码"""
        if not TF_AVAILABLE or self.model is None:
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': '模型不可用'
            }
        
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            if processed_image is None:
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': '图像预处理失败'
                }
            
            # 添加batch维度
            input_data = np.expand_dims(processed_image, axis=0)
            
            # 模型预测
            predictions = self.model.predict(input_data, verbose=0)
            
            # 解码预测结果
            decoded_text, confidence = self._decode_predictions(predictions[0])
            
            return {
                'success': True,
                'text': decoded_text,
                'confidence': confidence,
                'method': 'CRNN'
            }
            
        except Exception as e:
            self.logger.error(f"CRNN预测失败: {e}")
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }

    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """预处理图像用于CRNN模型"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                if self.config.MODEL_CHANNELS == 1:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    processed = gray
                else:
                    processed = image
            else:
                if self.config.MODEL_CHANNELS == 3:
                    processed = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    processed = image
            
            # 调整尺寸
            target_height, target_width = self.config.MODEL_INPUT_SIZE[::-1]  # (height, width)
            processed = cv2.resize(processed, (target_width, target_height))
            
            # 归一化
            processed = processed.astype(np.float32) / 255.0
            
            # 确保正确的维度
            if self.config.MODEL_CHANNELS == 1 and len(processed.shape) == 2:
                processed = np.expand_dims(processed, axis=-1)
            elif self.config.MODEL_CHANNELS == 3 and len(processed.shape) == 2:
                processed = np.stack([processed] * 3, axis=-1)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            return None

    def _decode_predictions(self, predictions: np.ndarray) -> Tuple[str, float]:
        """解码CTC预测结果"""
        try:
            # 获取最可能的路径
            input_length = predictions.shape[0]
            
            # 贪心解码
            decoded_indices = []
            last_idx = -1
            
            for i in range(input_length):
                # 获取当前时间步最可能的字符
                current_idx = np.argmax(predictions[i])
                
                # CTC解码规则：
                # 1. 去除blank字符（索引为num_classes-1）
                # 2. 去除连续重复字符
                if current_idx != self.num_classes - 1 and current_idx != last_idx:
                    decoded_indices.append(current_idx)
                    last_idx = current_idx
                elif current_idx != last_idx:
                    last_idx = current_idx
            
            # 转换为字符
            decoded_text = ''.join([self.idx_to_char.get(idx, '') for idx in decoded_indices])
            
            # 计算平均置信度
            max_probs = np.max(predictions, axis=1)
            confidence = np.mean(max_probs)
            
            return decoded_text, float(confidence)
            
        except Exception as e:
            self.logger.error(f"预测解码失败: {e}")
            return '', 0.0

    def train(self, train_data: List[Tuple[np.ndarray, str]], 
              val_data: List[Tuple[np.ndarray, str]] = None,
              epochs: int = 50, batch_size: int = 32) -> bool:
        """训练模型"""
        if not TF_AVAILABLE or self.model is None:
            self.logger.error("模型不可用")
            return False
        
        try:
            # 准备训练数据
            X_train, y_train = self._prepare_training_data(train_data)
            if X_train is None:
                return False
            
            # 准备验证数据
            if val_data:
                X_val, y_val = self._prepare_training_data(val_data)
                validation_data = (X_val, y_val) if X_val is not None else None
            else:
                validation_data = None
            
            # 设置回调函数
            callbacks = [
                ModelCheckpoint(
                    self.config.CRNN_MODEL_PATH,
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if validation_data else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("模型训练完成")
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return False

    def _prepare_training_data(self, data: List[Tuple[np.ndarray, str]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """准备训练数据"""
        try:
            X = []
            y = []
            
            for image, label in data:
                # 预处理图像
                processed_image = self._preprocess_image(image)
                if processed_image is None:
                    continue
                
                # 编码标签
                encoded_label = self._encode_label(label)
                if encoded_label is None:
                    continue
                
                X.append(processed_image)
                y.append(encoded_label)
            
            if not X:
                self.logger.error("没有有效的训练数据")
                return None, None
            
            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"准备了 {len(X)} 个训练样本")
            return X, y
            
        except Exception as e:
            self.logger.error(f"训练数据准备失败: {e}")
            return None, None

    def _encode_label(self, label: str) -> Optional[np.ndarray]:
        """编码标签为数字序列"""
        try:
            encoded = []
            for char in label:
                if char in self.char_to_idx:
                    encoded.append(self.char_to_idx[char])
                else:
                    self.logger.warning(f"未知字符: {char}")
                    return None
            
            # 填充到固定长度（用于批量训练）
            max_length = 8  # 车牌最大长度
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded.extend([self.num_classes - 1] * (max_length - len(encoded)))  # 用blank填充
            
            return np.array(encoded)
            
        except Exception as e:
            self.logger.error(f"标签编码失败: {e}")
            return None

    def evaluate_on_test_set(self, test_data: List[Tuple[np.ndarray, str]]) -> Dict[str, float]:
        """在测试集上评估模型"""
        if not test_data:
            return {'accuracy': 0.0, 'char_accuracy': 0.0}
        
        correct_plates = 0
        correct_chars = 0
        total_chars = 0
        
        for image, true_label in test_data:
            result = self.predict(image)
            predicted_label = result.get('text', '')
            
            # 整个车牌准确率
            if predicted_label == true_label:
                correct_plates += 1
            
            # 字符级准确率
            for i, (pred_char, true_char) in enumerate(zip(predicted_label, true_label)):
                if pred_char == true_char:
                    correct_chars += 1
                total_chars += 1
        
        plate_accuracy = correct_plates / len(test_data) if test_data else 0.0
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        
        return {
            'plate_accuracy': plate_accuracy,
            'char_accuracy': char_accuracy,
            'total_samples': len(test_data)
        }
