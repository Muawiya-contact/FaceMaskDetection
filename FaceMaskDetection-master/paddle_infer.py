import cv2
import argparse
import numpy as np
import os
import sys
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

# Disable oneDNN/MKLDNN before importing Paddle to avoid fused_conv2d oneDNN errors
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"

try:
    # Modern Paddle inference API
    from paddle.inference import Config as PaddleConfig
    from paddle.inference import create_predictor
    PADDLE_API = 'inference'
except Exception:
    try:
        # Older paddle (paddle.fluid) fallback
        import paddle.fluid as fluid
        from paddle.fluid.core import AnalysisConfig
        from paddle.fluid.core import create_paddle_predictor
        PaddleConfig = None
        create_predictor = None
        PADDLE_API = 'fluid'
    except Exception:
        PADDLE_API = None

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
colors = ((0, 255, 0), (0, 0 , 255))


def parse_source(value):
    if value == '0':
        return 0
    if value.isdigit():
        return int(value)
    return value


def is_image_source(value):
    _, ext = os.path.splitext(value.lower())
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def infer_frame(predictor, frame, target_shape=(260, 260)):
    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(img_rgb, target_shape)
    image_np = image_resized / 255.0
    image_np = image_np.transpose(2, 0, 1)
    input_data = np.expand_dims(image_np, axis=0).copy().astype('float32')

    if hasattr(predictor, "get_input_handle"):
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.copy_from_cpu(input_data)
        predictor.run()
        output_names = predictor.get_output_names()
        y_bboxes_output = predictor.get_output_handle(output_names[0]).copy_to_cpu()
        y_cls_output = predictor.get_output_handle(output_names[1]).copy_to_cpu()
    else:
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_tensor(input_names[0])
        input_tensor.copy_from_cpu(input_data)
        predictor.zero_copy_run()
        output_names = predictor.get_output_names()
        y_bboxes_output = predictor.get_output_tensor(output_names[0]).copy_to_cpu()
        y_cls_output = predictor.get_output_tensor(output_names[1]).copy_to_cpu()

    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=0.5, iou_thresh=0.4)

    # Draw on original BGR frame (not the preprocessed tensor)
    show = frame.copy()
    tl = round(0.002 * (height + width) * 0.5) + 1
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        cv2.rectangle(show, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
        cv2.putText(show, "%s: %.2f" % (id2class[class_id], conf),
                    (xmin + 2, ymin - 2), 3, 0.8, colors[class_id])
    return show

    if hasattr(predictor, "get_input_handle"):
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.copy_from_cpu(img)
        predictor.run()
        output_names = predictor.get_output_names()
        y_bboxes_output = predictor.get_output_handle(output_names[0]).copy_to_cpu()
        y_cls_output = predictor.get_output_handle(output_names[1]).copy_to_cpu()
    else:
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_tensor(input_names[0])
        input_tensor.copy_from_cpu(img)
        predictor.zero_copy_run()
        output_names = predictor.get_output_names()
        y_bboxes_output = predictor.get_output_tensor(output_names[0]).copy_to_cpu()
        y_cls_output = predictor.get_output_tensor(output_names[1]).copy_to_cpu()

    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=0.5, iou_thresh=0.4)

    tl = round(0.002 * (height + width) * 0.5) + 1
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
        cv2.putText(img, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2), 3, 0.8, colors[class_id])

    return img

def load_model(model_file, params_file, use_gpu=False, use_mkl=False, mkl_thread_num=4):
    """
    Load a Paddle predictor. Supports both `paddle.inference` and `paddle.fluid` APIs.
    """
    if PADDLE_API == 'inference':
        config = PaddleConfig(model_file, params_file)
        if use_gpu:
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        # do not enable mkldnn by default; only if explicitly requested
        if use_mkl and not use_gpu:
            try:
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(mkl_thread_num)
            except Exception:
                pass
        # keep IR optim off to avoid fused_conv2d issues
        try:
            config.disable_glog_info()
            config.enable_memory_optim()
            config.switch_ir_optim(False)
        except Exception:
            pass
        predictor = create_predictor(config)
    elif PADDLE_API == 'fluid':
        config = fluid.core.AnalysisConfig(model_file, params_file)
        if use_gpu:
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        if use_mkl and not use_gpu:
            try:
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(mkl_thread_num)
            except Exception:
                pass
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_ir_optim(False)
        config.switch_use_feed_fetch_ops(False)
        predictor = fluid.core.create_paddle_predictor(config)
    else:
        raise RuntimeError('Paddle not available in this environment')
    return predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--model_dir', type=str, default='models/paddle', help='model path')
    parser.add_argument('--source', type=str, default='0', help='camera index like 0, or a path to an image/video file')
    parser.add_argument('--use-mkldnn', action='store_true', help='Enable MKL-DNN (oneDNN) optimizations')
    args = parser.parse_args()

    # detect model files in `models/paddle/` (support both legacy __model__/__params__ and new .pdmodel/.pdiparams)
    model_dir = args.model_dir
    model_file = None
    params_file = None
    if os.path.isdir(model_dir):
        # legacy files
        legacy_model = os.path.join(model_dir, '__model__')
        legacy_params = os.path.join(model_dir, '__params__')
        if os.path.exists(legacy_model) and os.path.exists(legacy_params):
            model_file, params_file = legacy_model, legacy_params
        else:
            # try new single-file format .pdmodel/.pdiparams
            for f in os.listdir(model_dir):
                if f.endswith('.pdmodel'):
                    model_file = os.path.join(model_dir, f)
                if f.endswith('.pdiparams') or f.endswith('.pdparams'):
                    params_file = os.path.join(model_dir, f)
    # fallback: try top-level paths if user passed the model file directly
    if model_file is None or params_file is None:
        # try default names adjacent to model_dir path
        if os.path.exists(model_dir + '.__model__') and os.path.exists(model_dir + '.__params__'):
            model_file, params_file = model_dir + '.__model__', model_dir + '.__params__'

    if model_file is None or params_file is None:
        # If paddle files not found, provide fallback to OpenCV DNN
        print('Paddle model files not found in', model_dir)
        print('Falling back to OpenCV DNN (requires .prototxt + .caffemodel in models/)')
        # Try to locate caffe model
        proto = os.path.join('models', 'face_mask_detection.prototxt')
        caffemodel = os.path.join('models', 'face_mask_detection.caffemodel')
        if os.path.exists(proto) and os.path.exists(caffemodel):
            print('Using OpenCV DNN fallback')
            net = cv2.dnn.readNet(caffemodel, proto)
            from opencv_dnn_infer import run_on_video
            run_on_video(net, 0, conf_thresh=0.5)
            sys.exit(0)
        else:
            raise FileNotFoundError('No Paddle model and no Caffe model found. Place model files in models/paddle or models/')

    predictor = load_model(model_file, params_file, use_mkl=args.use_mkldnn)
    source = parse_source(args.source)

    if isinstance(source, str) and is_image_source(source):
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError('Could not read image source: %s' % source)
        image = infer_frame(predictor, image)
        cv2.imshow("img", image)
        cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError('Video open failed.')
        while True:
            ret, img = cap.read()
            if not ret:
                break
            image = infer_frame(predictor, img)
            cv2.imshow("img", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break