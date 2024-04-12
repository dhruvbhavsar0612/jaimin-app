from keras.models import load_model

model = tf.keras.models.load_model('my_model2.hdf5')


def preprocess_image(image):
    img = image.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Make predictions on the input image
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions)
    class_label = ['corn_earworm', 'fall_armyworm'][class_index]
    confidence = predictions[0][class_index] * 100
    return class_label, confidence