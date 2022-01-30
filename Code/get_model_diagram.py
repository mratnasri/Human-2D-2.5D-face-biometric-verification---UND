import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import visualkeras
from PIL import ImageFont
#model = load_model('../Models/human_2D_model21_dropout.h5')
model = load_model('../Models/human_2D_verification_model9_DA_3.h5')
model.summary()
# keras.utils.plot_model(model,"../../Outputs/human_2D_model21_dropout_architecture.png",show_shapes=True)
#font = ImageFont.truetype("arial.ttf", 32)
'''visualkeras.layered_view(
    model, scale_z=0.3, scale_xy=2, legend=True, to_file="../../Outputs/human_2D_verification_model9_DA_3_architecture_visualkeras_legend.png")'''
visualkeras.graph_view(
    model, to_file="../../Outputs/human_2D_verification_model9_DA_3_architecture_visualkeras_graph_legend.png")
