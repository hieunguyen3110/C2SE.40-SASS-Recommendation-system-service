from flask import jsonify,abort

class ApiResponse:
    @staticmethod
    def success(data=None, message="success", code=200):
        return jsonify({
            "code": code,
            "message": message,
            "data": data
        })

    @staticmethod
    def error(message="Something went wrong", code=500):
        return abort(code, {
            "code": code,
            "message": message,
            "data": None
        })
