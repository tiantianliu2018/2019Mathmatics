import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd

import pickle

class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        file_data = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                input_data = np.array(pb_data.get_values()[:, 0:17], dtype=np.float32)
                print(file_name, input_data.shape)
                file_data.append(input_data)

        file_data = np.array(file_data, dtype=np.float32).reshape(-1, 17)

        preprocessed_data['myInput'] = file_data
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        print(self.model_path)

        output = self.my_process(preprocessed_data)
        print("yubzhu successfully insert data to model!")
        return output

    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output

    def my_process(self, data):
        data = pd.DataFrame(data['myInput'])
        data.columns = ['Cell Index', 'Cell X', 'Cell Y', 'Height', 'Azimuth', 'Electrical Downtilt',
                        'Mechanical Downtilt',
                        'Frequency Band', 'RS Power', 'Cell Altitude', 'Cell Building Height', 'Cell Clutter Index',
                        'X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']
        # change below

        ## 
        data['relative_x'] = (data['X']-data['Cell X'])
        data['relative_y'] = (data['Y']-data['Cell Y'])
        data['distance'] = (data['relative_x']**2 + data['relative_y']**2)**0.5
        data.drop(['relative_x','relative_y'],axis = 1,inplace=True)
        ## 
        data['angle'] = data['Electrical Downtilt'] + data['Mechanical Downtilt']
        ## 
        data['relative_altitude'] = data['Altitude'] - data['Cell Altitude']

        data['deltaHv'] = data['Height'] - np.tan(data['angle']/180.0 * np.pi) * data['distance'] + data['Cell Altitude'] - data['Altitude']
        data['h_theta'] = 90 - np.arctan2((data['Y']-data['Cell Y']), (data['X']-data['Cell X']))/np.pi * 180.0 - data['Azimuth']
        data['h_distance'] = 2*abs(np.sin(data['h_theta']/np.pi*180.0/2)) * data['distance']

        ## 
        data.loc[data['Height'] == 0,['Height']] = 22.0
        ## 
        data.loc[data['distance'] == 0,['distance']] = 1
        data.loc[data['h_distance'] == 0,['h_distance']] = 1
        ## 
        data['Height'] = np.log10(data['Height'])
        data['Frequency Band'] = np.log10(data['Frequency Band'])
        data['distance'] = np.log10(data['distance'])
        data['h_distance'] = np.log10(data['h_distance'])

        
        data['angle_level']=np.arctan2(data['X']-data['Cell X'],data['Y']-data['Cell Y'])*180/np.pi
        data['angle_level']=np.abs(data['Azimuth']-data['angle_level'])
        data['angle_level']= np.where(data['angle_level']<=180, data['angle_level'], 360.0-data['angle_level'])

        data['hd'] = data['distance'] * data['Height']
        data['f_sub_h'] = data['Frequency Band']-data['Height']
        data['f_add_d'] = data['Frequency Band']+data['distance']
        data['f_sub_hd'] = data['Frequency Band']-data['hd']

        data['f_sub_h_add_d'] = data['f_sub_h']+data['distance']
        data['f_sub_h_sub_hd'] = data['f_sub_h']-data['hd']
        data['f_add_d_sub_hd'] = data['f_add_d']-data['hd']

        data['f_sub_h_add_d_sub_hd'] = data['f_sub_h_add_d'] - data['hd']
        print(data.shape)
        data = data.drop(['Cell Index','Cell X', 'Cell Y','Azimuth','Electrical Downtilt','Mechanical Downtilt','Cell Altitude','Cell Building Height','angle','X','Y','Altitude'],axis=1)
  
        print(data.shape)
        fr = open(self.model_path + '/misc/model_ltt', 'rb')
        model = pickle.load(fr)
        output = model.predict(data)
        output = np.array(output, dtype=np.float32).reshape(-1, 1)
        print(output.shape)
        output_dict = {}
        output_dict['myInput'] = output
        return output_dict
