from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SputumCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.model = self._build_model(input_shape, num_classes)

    def _build_model(self, input_shape, num_classes):
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_dir, validation_dir, epochs=50, batch_size=32):
        train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical'
        )
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical'
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            'data/models/best_ocr_model.keras', monitor='val_loss', save_best_only=True, mode='min'
        )
        history = self.model.fit(
            train_generator, steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs, validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=[early_stopping, model_checkpoint]
        )
        self.model.save('data/models/final_ocr_model.keras')
        return history

    def evaluate(self, test_dir, batch_size=32):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical'
        )
        loss, accuracy = self.model.evaluate(test_generator)
        return loss, accuracy