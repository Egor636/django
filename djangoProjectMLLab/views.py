from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image, ImageOps
import tensorflow as tf
import os

classes = ['американский бульдог', 'американский питбуль терьер', 'бассет-хаунд', 'бигль', 'боксер', 'чихуахуа',
           'английский кокер-спаниель', 'английский сеттер', 'немецкий курцхаар', 'великий пиреней', 'хаванез',
           'японский хин', 'кизхонд', 'леонбергер', 'миниатюрный пинчер', 'ньюфаундленд', 'померанский', 'мопс',
           'сенбернар', 'самоед', 'шотландский терьер', 'шиба ину', 'стаффордширский бультерьер', 'уитен терьер',
           'йоркширский терьер']

model = tf.keras.models.load_model('djangoProjectMLLab/dogs_v3.h5')

def upload_file(request):
    if request.method == 'POST' and request.FILES:
        file = request.FILES['myfile']
        upload_folder = os.path.join('djangoProjectMLLab', 'uploaded_images')
        fs = FileSystemStorage(location=upload_folder)

        filename = fs.save(file.name, file)
        file_url = fs.url(filename)


        size = (332, 332)
        image = Image.open(file)
        image = image.convert("RGB")
        image = ImageOps.fit(image, size, Image.LANCZOS)
        img_array = tf.keras.preprocessing.image.img_to_array(image)[tf.newaxis, ...]

        predictions = model.predict(img_array)
        predicted_class_index = tf.argmax(predictions, axis=1)[0]


        res_message = f'Скорее всего на этом фото {classes[predicted_class_index]}'

        return render(request, 'upload_file.html', {
            'res_message': res_message,
        })
    return render(request, 'upload_file.html')
