from flask import Flask, render_template, send_file
import os
import io
import sys

#changing directory to get the heatmap.py functions
#it is not being cloned into the server dockerfile

#current_dir = os.path.dirname(os.path.abspath(__file__))
#src_dir = os.path.join(current_dir, 'chocolatechip','src')
##print(src_dir)
#from chocolatechip.MySQLConnector import MySQLConnector

# from flask import request, jsonify

app = Flask(__name__)

@app.route('/') #localhost
def home():
    return render_template('home.html')

@app.route('/cameras') #localhost/cameras
def cameras():
    cameras = [{'number':'1','intersection':'Alachua'},
               {'number':'2','intersection':'Broward'}]
    return render_template('cameras.html', author = "Artur", cam_active = True, cameras = cameras)

@app.route('/cameras/<camera_id>')
def cameranum(camera_id):
    return render_template('cameras_num.html', camera_id = str(camera_id))

# 3287 intersec_id
# 'track' df_type
# mean = true

# @app.route('/heatmap', methods=['GET'])
# def display_heatmap():
#     intersec_id = int(request.args.get('intersec_id')) 
#     df_type = request.args.get('df_type')
#     mean = request.args.get('mean', 'True') == 'True'

#     try:
#         base64_img = heatmap_generator(df_type, mean, intersec_id)
        
#         return jsonify({'image': base64_img}) # return img, not sure if its data will disp or the img itself
#     except Exception:
#         return jsonify({'error': 'An error occurred'}), 500

@app.route('/image_generator')
def display_generated_image_in_memory():
    binary_data = heatmap.heatmap_generator(
        df_type="track",
        mean=True,
        intersec_id=3252, #testing with intersection 3252
        p2v=False,
        conflict_type=None,
        pedestrian_counting=False,
        return_agg=True
    )
    return send_file(io.BytesIO(binary_data), mimetype='image/png')


#if __name__ == "__main__":
#    app.run()

#to run with gunicorn gunicorn -w 4 -b 0.0.0.0:8000 server:app
#pkill gunicorn to kill processes related to gunicorn (might be dangerou)