from common.exceptions.base import BaseExc

from agent.exceptions.codes import c_40600, c_40601, c_40602, c_40603, c_40604


class BotExc(BaseExc):
    pass


BotProtocolSynchronizationFailedExc = BotExc(*c_40600)
BotPublishFailedExc = BotExc(*c_40601)
BotAuthFailedExc = BotExc(*c_40602)
BotNotFoundExc = BotExc(*c_40603)
TenantNotFoundExc = BotExc(*c_40604)
