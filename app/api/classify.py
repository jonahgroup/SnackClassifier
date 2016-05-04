import json

from flask import (
    request,
    Response
)

from . import api
from ..image_utils import (
    get_trainer,
    get_image_classify,
    get_image_class_names,
    save_trained_image
)


"""
@api {get} /classify Get images
@apiVersion 1.0.0
@apiName GetClassifyImage
@apiDescription This call returns the classified image .
"""


@api.route('/classify', methods=['GET'])
def get_classify():
    image_url = request.args.get('image')
    data = get_image_classify(get_trainer(), image_url)
    return Response(json.dumps(data), mimetype='application/json')

"""
@api {post} /classify Post a list of images
@apiVersion 1.0.0
@apiName GetClassifyImages
@apiDescription This call returns the list of classified image .
"""


@api.route('/classify', methods=['POST'])
def post_classify():
    result = {}
    image_url_list = request.json
    if image_url_list is not None:
        result_list = []
        for image_url in image_url_list['images']:
            data = get_image_classify(get_trainer(), image_url)
            result_list.append(data)
        result['images'] = result_list
    return Response(json.dumps(result), mimetype='application/json')

"""
@api {get} /classify/names Get list of class names
@apiVersion 1.0.0
@apiName GetClassNames
@apiDescription This call returns a list of the possible class names that an image can be classified by.

@apiSuccess {str[]} strs The list of class names
"""


@api.route('/classify/names', methods=['GET'])
def get_classify_names():
    name_list = get_image_class_names(get_trainer())
    return Response(json.dumps(name_list), mimetype='application/json')


"""
@api {post} /train Post a list of trained images
@apiVersion 1.0.0
@apiName PostTrainedImages
@apiDescription This call save the list of trained image .
"""


@api.route('/train', methods=['POST'])
def post_train():
    result = {}
    trained_image_list = request.json
    if trained_image_list is not None:
        result_list = []
        for trained_image in trained_image_list['images']:
            data = save_trained_image(trained_image['class'], trained_image['image'])
            result_list.append(data)
        result['images'] = result_list
    return Response(json.dumps(result), mimetype='application/json')


"""
@api {get} /retrain Trigger retraining from the existed training set
@apiVersion 1.0.0
@apiName GetRetrain
@apiDescription This call trigger retraining and return a list of trained classes
"""


@api.route('/retrain', methods=['GET'])
def get_retrain():
    result = {}
    trainer = get_trainer(retrain=True)
    if trainer is not None:
        name_list = get_image_class_names(get_trainer())
        return Response(json.dumps(name_list), mimetype='application/json')
    else:
        return Response(json.dumps(result), status=400, mimetype='application/json')
