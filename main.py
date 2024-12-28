import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

def load_data(start = 0):
    data = pd.read_csv('list_bbox_celeba.csv')
    
    images = []
    bboxes = []
    
    for i in range(start, start + 2500):
        # Load image without resizing
        img = cv2.imread('img_align_celeba/' + data['image_id'][i])
        if img is not None:
            # Normalize pixel values to [0, 1]
            img = img / 255.0
            images.append(img)
            
            height, width = img.shape[:2]
            bbox = np.array([
                data['x_1'][i] / width,
                data['y_1'][i] / height,
                data['width'][i] / width,
                data['height'][i] / height
            ], dtype=np.float32)
            bboxes.append(bbox)
    
    return np.array(images), np.array(bboxes)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=200,
        decay_rate=0.9,
        staircase=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.Huber(),
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # Load the data
    images, bboxes = load_data()
    
    # Print shapes to verify
    print("Images shape:", images.shape)
    print("Bboxes shape:", bboxes.shape)
    
    # Build and train model with actual image dimensions
    if len(images) > 0:
        input_shape = images[0].shape
        model = build_model(input_shape)
        model.fit(images, bboxes, epochs=5, batch_size=32)

    # test last 1000 images from set
    test_images, test_bboxes = load_data(2000)

    model.evaluate(test_images, test_bboxes)

    # Save model
    model.save('model.h5')
    