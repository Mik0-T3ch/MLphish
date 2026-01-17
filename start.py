print('hola mundo')

from src.model import MLPhishModel
#aca crearia el modelo pero hay un problema y es que se necesita un argumento para input_dim
modelo =  MLPhishModel()

#el link pishing
link = input('link: ')

#cargo el modelo ya entrenado
modelo.load_trained_model()

#le doy el link al modelo
resultado = modelo.predict(link)

#imprimo el resultado del modelo sobre el link sospechoso
print(f'respuesta: {resultado}')
