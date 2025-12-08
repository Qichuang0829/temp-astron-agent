import json
import os
from typing import Annotated
from datetime import datetime

import aiohttp
from fastapi import APIRouter, Header

from agent.api.schemas.bot_manage_inputs import ProtocolSynchronization, Publish, Auth
from agent.api.schemas.bot_manage_response import BotResponse, build_bot_response
from common.service import get_db_service
from common.otlp.trace.span import Span
from common.service.db.db_service import session_getter
from agent.domain.models.bot import Bot, BotRelease
from agent.infra.app_auth import APPAuth
from agent.exceptions.bot_exc import (
    BotExc,
    BotProtocolSynchronizationFailedExc,
    BotPublishFailedExc,
    BotAuthFailedExc,
    BotNotFoundExc,
    AppAuthFailedExc,
)
from agent.exceptions.codes import c_0

bot_manage_router = APIRouter()


@bot_manage_router.post("/bot")  # type: ignore[misc]
async def protocol_synchronization(x_consumer_username: Annotated[str, Header()], inputs: ProtocolSynchronization) -> BotResponse:
    """Protocol Synchronization - 协议同步：根据ID查询数据，存在则更新，不存在则创建"""
    error: BotExc = BotExc(*c_0)
    span = Span(
        app_id=x_consumer_username,
    )
    with span.start("ProtocolSynchronization") as sp:
        try:
            sp.set_attribute("id", inputs.id)
            sp.add_info_events({"protocol-synchronization-inputs": inputs.model_dump_json(by_alias=True)})

            app_detail = await APPAuth().app_detail(x_consumer_username)
            sp.add_info_events(
                {"kong-app-detail": json.dumps(app_detail, ensure_ascii=False)}
            )
            if app_detail is None:
                raise AppAuthFailedExc("Cannot find appid authentication information")
            if app_detail.get("code") != 0:
                raise AppAuthFailedExc(app_detail.get("message", ""))

            with session_getter(get_db_service()) as session:
                # 根据 id 查询是否存在该记录
                existing_bot = session.query(Bot).filter(Bot.id == inputs.id).first()

                # 将 dsl 对象转换为 JSON 字符串
                dsl_json = json.dumps(inputs.dsl.model_dump(exclude_none=True), ensure_ascii=False) if inputs.dsl else None

                # 使用 x_consumer_username 作为 app_id
                app_id = x_consumer_username
                current_time = datetime.now()

                if existing_bot:
                    # 如果记录存在，更新数据
                    if dsl_json:
                        existing_bot.dsl = dsl_json
                    existing_bot.app_id = app_id
                    existing_bot.update_at = current_time
                    session.add(existing_bot)
                    session.commit()
                    session.refresh(existing_bot)
                    bot = existing_bot
                else:
                    # 如果记录不存在，创建新记录
                    # pub_status 默认值为 0（草稿状态）
                    new_bot = Bot(
                        id=inputs.id,
                        app_id=app_id,
                        dsl=dsl_json or "",
                        pub_status=0,  # 默认状态为草稿
                        create_at=current_time,
                        update_at=current_time
                    )
                    session.add(new_bot)
                    session.commit()
                    session.refresh(new_bot)
                    bot = new_bot

                # 构建返回数据
                response_data = {
                    "id": bot.id,
                    "app_id": bot.app_id,
                    "pub_status": bot.pub_status,
                    "create_at": bot.create_at.isoformat() if bot.create_at else None,
                    "update_at": bot.update_at.isoformat() if bot.update_at else None
                }

                return build_bot_response(error, response_data)
        except BotExc as e:
            error = e
        except Exception as e:  # pylint: disable=broad-exception-caught
            error = BotProtocolSynchronizationFailedExc(str(e))
    
    return build_bot_response(error)


@bot_manage_router.post("/bot/publish")  # type: ignore[misc]
async def publish(x_consumer_username: Annotated[str, Header()], inputs: Publish) -> BotResponse:
    """发布助手：如果传入dsl则使用传入的dsl创建BotRelease，否则从Bot表查询并创建BotRelease，同时更新Bot的pub_status为1"""
    error: BotExc = BotExc(*c_0)
    span = Span(
        app_id=x_consumer_username,
    )
    with span.start("Publish") as sp:
        try:
            sp.set_attribute("bot_id", inputs.bot_id)
            sp.set_attribute("version", inputs.version)
            sp.add_info_events({"publish-inputs": inputs.model_dump_json(by_alias=True)})

            app_detail = await APPAuth().app_detail(x_consumer_username)
            sp.add_info_events(
                {"kong-app-detail": json.dumps(app_detail, ensure_ascii=False)}
            )
            if app_detail is None:
                raise AppAuthFailedExc("Cannot find appid authentication information")
            if app_detail.get("code") != 0:
                raise AppAuthFailedExc(app_detail.get("message", ""))

            with session_getter(get_db_service()) as session:
                current_time = datetime.now()

                # 判断是否传入了 dsl 字段
                if inputs.dsl is not None:
                    # 如果传入了 dsl，使用用户传入的 dsl 创建 BotRelease
                    dsl_json = json.dumps(inputs.dsl.model_dump(exclude_none=True), ensure_ascii=False)

                    # 创建 BotRelease 记录
                    bot_release = BotRelease(
                        bot_id=inputs.bot_id,
                        version=inputs.version,
                        description=inputs.description,
                        dsl=dsl_json,
                        create_at=current_time,
                        update_at=current_time
                    )
                    session.add(bot_release)
                    session.commit()
                    session.refresh(bot_release)

                    return build_bot_response(error, {"version": bot_release.id})
                else:
                    # 如果没有传入 dsl，使用 bot_id 查询 Bot 表中的数据
                    bot = session.query(Bot).filter(Bot.id == inputs.bot_id).first()

                    if not bot:
                        raise BotNotFoundExc(f"Bot with id {inputs.bot_id} not found")

                    # 使用 Bot 表中的 dsl 创建 BotRelease
                    bot_release = BotRelease(
                        bot_id=inputs.bot_id,
                        version=inputs.version,
                        description=inputs.description,
                        dsl=bot.dsl,  # 使用 Bot 表中的 dsl
                        create_at=current_time,
                        update_at=current_time
                    )
                    session.add(bot_release)

                    # 更新 Bot 表中的 pub_status 为 1（已发布）
                    bot.pub_status = 1
                    bot.update_at = current_time
                    session.add(bot)

                    session.commit()
                    session.refresh(bot_release)
                    session.refresh(bot)

                    return build_bot_response(error, {"version": bot_release.id})
        except BotExc as e:
            error = e
        except Exception as e:  # pylint: disable=broad-exception-caught
            error = BotPublishFailedExc(str(e))
    
    return build_bot_response(error)


@bot_manage_router.post("/bot/auth")  # type: ignore[misc]
async def auth(x_consumer_username: Annotated[str, Header()], inputs: Auth) -> BotResponse:
    """绑定能力：查询是否已绑定，如果未绑定则进行绑定操作"""
    error: BotExc = BotExc(*c_0)
    span = Span(
        app_id=x_consumer_username,
    )
    with span.start("Auth") as sp:
        try:
            # 获取环境变量
            auth_get_api_url = os.getenv("AUTH_GET_API_URL")
            auth_add_api_url = os.getenv("AUTH_ADD_API_URL")
            auth_required_username = os.getenv("AUTH_REQUIRED_USERNAME")

            sp.set_attribute("version_id", inputs.version_id)
            sp.set_attribute("app_id", inputs.app_id)
            sp.add_info_events({"auth-inputs": inputs.model_dump_json(by_alias=True)})

            app_detail = await APPAuth().app_detail(x_consumer_username)
            sp.add_info_events(
                {"kong-app-detail": json.dumps(app_detail, ensure_ascii=False)}
            )
            if app_detail is None:
                raise AppAuthFailedExc("Cannot find appid authentication information")
            if app_detail.get("code") != 0:
                raise AppAuthFailedExc(app_detail.get("message", ""))

            # 构建查询参数
            params = {
                "app_id": inputs.app_id,
                "type": "agent",
                "ability_id": str(inputs.version_id)
            }

            # 查询是否已经绑定
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=5)
                try:
                    async with session.get(
                        auth_get_api_url,
                        params=params,
                        timeout=timeout
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()

                        # 第三方返回非成功 code，认为查询失败
                        if result.get("code") != 0:
                            raise BotAuthFailedExc(
                                f"Failed to query binding status from third-party, "
                                f"code={result.get('code')}, message={result.get('message', '')}"
                            )

                        # 判断是否已经绑定（如果 data 数组不为空，说明已绑定）
                        if result.get("data") and len(result.get("data", [])) > 0:
                            return build_bot_response(
                                error, None, "该能力 id 已绑定，无需重新绑定"
                            )
                except aiohttp.ClientError as e:
                    raise BotAuthFailedExc(
                        f"Failed to query binding status: {str(e)}"
                    )

                # 如果没有绑定，进行绑定操作
                request_body = {
                    "app_id": inputs.app_id,
                    "type": "agent",
                    "ability_id": str(inputs.version_id)
                }

                headers = {
                    "x-consumer-username": auth_required_username,
                    "Content-Type": "application/json"
                }

                try:
                    async with session.post(
                        auth_add_api_url,
                        json=request_body,
                        headers=headers,
                        timeout=timeout
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()

                        # 判断是否绑定成功
                        if result.get("code") == 0:
                            return build_bot_response(error, None, "绑定成功")
                        else:
                            raise BotAuthFailedExc(result.get("message", "绑定失败"))
                except aiohttp.ClientError as e:
                    raise BotAuthFailedExc(f"Failed to bind: {str(e)}")
        except BotExc as e:
            error = e
        except Exception as e:  # pylint: disable=broad-exception-caught
            error = BotAuthFailedExc(str(e))
    
    return build_bot_response(error)

