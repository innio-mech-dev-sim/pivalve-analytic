if __name__ == '__main__':
    import os

    os.chdir('../')

from typing import Dict, NewType, Tuple
import logging
import pandas as pd

from config import config
from rms_commons.constants.authenticator import Authenticator

pd.set_option('mode.chained_assignment', None)
Token = NewType('Token', str)

logger = logging.getLogger('app-logger')


class DefaultAuthenticatorInstance:
    _authenticator = Authenticator(
        url=f"{config['MYPLANT_URL']}/api/oauth/token",
        api_id=config["MYPLANT_APP_ID"],
        api_secret=config["MYPLANT_APP_SECRET"]
    )

    @staticmethod
    def get_authenticator() -> Authenticator:
        return DefaultAuthenticatorInstance._authenticator


auth = DefaultAuthenticatorInstance.get_authenticator()
