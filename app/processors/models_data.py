models_dir = './models'
assets_repo = "https://github.com/visomaster/visomaster-assets/releases/download"

try:
    import tensorrt as trt
    models_trt_list = []
except ModuleNotFoundError:
    models_trt_list = []

arcface_mapping_model_dict = {
    'Inswapper128': 'Inswapper128ArcFace',
}

detection_model_mapping = {
    'RetinaFace': 'RetinaFace',
}

landmark_model_mapping = {

}



models_list = [
    {
        "model_name": "Inswapper128",
        "local_path": f"{models_dir}/inswapper_128.fp16.onnx",
        "hash": "6d51a9278a1f650cffefc18ba53f38bf2769bf4bbff89267822cf72945f8a38b",
        "url": f"{assets_repo}/v0.1.0/inswapper_128.fp16.onnx"
    },
    {
        "model_name": "RetinaFace",
        "local_path": f"{models_dir}/det_10g.onnx",
        "hash": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        "url": f"{assets_repo}/v0.1.0/det_10g.onnx"
    },
    {
        "model_name": "Inswapper128ArcFace",
        "local_path": f"{models_dir}/w600k_r50.onnx",
        "hash": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "url": f"{assets_repo}/v0.1.0/w600k_r50.onnx"
    },
    {
        "model_name": "GFPGANv1.4",
        "local_path": f"{models_dir}/GFPGANv1.4.onnx",
        "hash": "6548e54cbcf248af385248f0c1193b359c37a0f98b836282b09cf48af4fd2b73",
        "url": f"{assets_repo}/v0.1.0/GFPGANv1.4.onnx"
    },
    {
        "model_name": "XSeg",
        "local_path": f"{models_dir}/XSeg_model.onnx",
        "hash": "4381395dcbec1eef469fa71cfb381f00ac8aadc3e5decb4c29c36b6eb1f38ad9",
        "url": f"{assets_repo}/v0.1.0/XSeg_model.onnx"
    } 
]