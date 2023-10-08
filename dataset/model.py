from ultralytics import YOLO
	
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
 
model.train(data="custom.yaml",epochs=6,imgsz=480,batch=8)  # train the model
model.val()  # evaluate model performance on the validation set
model.predict(source="https://img.freepik.com/premium-photo/ripe-mango-with-green-leaf-isolated-white_252965-183.jpg")  # predict on an image
model.export(format="onnx")  