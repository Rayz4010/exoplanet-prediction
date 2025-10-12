import pickle as pkl

model=pkl.load(open('exoplanet_model.pkl', 'rb'))

data=[0.000,19.899140,1.490000e-05,-1.490000e-05,175.850252,0.000581,-0.000581,0.969,5.126,-0.077,1.78220,0.03410,-0.03410,10800.0,171.0,-171.0,14.60,3.92,-1.31,638.0,39.30,31.04,-10.49,76.3,1.0,5853.0,158.0,-176.0,4.544,0.044,-0.176,0.868,0.233,-0.078,297.00482,48.134129,15.436]

pred=model.predict([data])
if pred==0:
    print("False Positive")
elif pred==1:
    print("Exoplanet Detected")
else:
    print("Exoplanet Confirmed")
    