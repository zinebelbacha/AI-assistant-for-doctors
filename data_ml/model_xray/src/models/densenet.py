from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logging import setup_logging
import os

class XRayDenseNet:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, model_path=None):
        self.logger = setup_logging()
        self.input_shape = input_shape
        self.num_classes = num_classes
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            self.logger.info(f"Loaded model from {model_path}")
        else:
            self.model = self._build_model()
            self.logger.info("Built new DenseNet model")

    def _build_model(self):
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, validation_dir, epochs=50, batch_size=32):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=self.input_shape[:2], batch_size=batch_size, class_mode='categorical'
        )
        val_generator = val_datagen.flow_from_directory(
            validation_dir, target_size=self.input_shape[:2], batch_size=batch_size, class_mode='categorical'
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            os.path.join('data/models', 'best_densenet_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch_size,
            callbacks=[early_stopping, checkpoint]
        )
        self.model.save(os.path.join('data/models', 'final_densenet_model.keras'))
        self.logger.info("Model saved as 'final_densenet_model.keras'")
        return history

    def evaluate(self, test_dir, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=self.input_shape[:2], batch_size=batch_size, class_mode='categorical'
        )
        loss, accuracy = self.model.evaluate(test_generator)
        self.logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy