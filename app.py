import rembg
import numpy as np
from PIL import Image
import os
import io
from flask import Flask, request, send_file
import requests
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
print("Servicio iniciado")

directory = r"D:\imagenes_sin_fondo"
os.makedirs(directory, exist_ok=True)
print(f'Directorio "{directory}" creado o ya existe.')

@app.route('/fondo-blanco', methods=['POST'])
def fondo_blanco():
    if 'file' not in request.files:
        return jsonify({'mensaje': 'Error: No se encontró el archivo.'}), 400

    session = rembg.new_session("u2netp")
    file = request.files['file']
    name_salida = file.filename

    try:
        input_image = Image.open(file)
    except Exception as e:
        return jsonify({'mensaje': f'Error al abrir la imagen: {str(e)}'}), 400

    input_array = np.array(input_image)
    output_array = rembg.remove(input_array, alpha_matting=True, alpha_matting_erode_size=10)
    output_image = Image.fromarray(output_array)

    result = Image.new("RGB", input_image.size, 'white')
    result.paste(output_image, mask=output_image)

    # Guardar como archivo PNG
    name_salida = os.path.join(directory, "sin_fondo_" + os.path.splitext(name_salida)[0] + ".png")
    result.save(name_salida, 'PNG')

    # Devolver el archivo como respuesta
    return send_file(name_salida, mimetype='image/png', as_attachment=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9566)