class autoMLSession:

    parameters = {}
    is_cleared = False
    is_configured = False

    @classmethod
    def add_session_parameter(cls, value:any, key:str):
        cls.parameters[key] = value      
        cls.is_cleared = False


    @classmethod
    def get_session_parameter(cls, key:str):
        if key in cls.parameters:
            return cls.parameters[key]
        else:
            return None

    @classmethod
    def clear_session(cls):
        cls.parameters = {}
        cls.is_cleared = True
        cls.is_configured = False
