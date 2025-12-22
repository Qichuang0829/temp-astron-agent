import json
import os
from datetime import datetime
from typing import Annotated, Union

import aiohttp
from fastapi.openapi.models import APIKey
from common.otlp.trace.span import Span
from common.service import get_db_service
from common.service.db.db_service import session_getter
from fastapi import APIRouter, Header

from agent.api.schemas_v2.bot_manage_inputs import (
    Auth,
    ProtocolSynchronization,
    Publish,
)
from agent.api.schemas_v2.bot_manage_response import BotResponse, build_bot_response
from agent.domain.models.bot import AppAuthDetail, Bot, BotRelease, BotTenant
from agent.exceptions.bot_exc import (
    AppAuthFailedExc,
    BotAuthFailedExc,
    BotExc,
    BotNotFoundExc,
    BotProtocolSynchronizationFailedExc,
    BotPublishDuplicatedExc,
    BotPublishFailedExc,
    TenantNotFoundExc,
)
from agent.exceptions.codes import c_0
from agent.infra.app_auth import APPAuth

bot_manage_router = APIRouter()


async def _validate_tenant(x_consumer_username: str) -> None:
    """Validate tenant information"""
    with session_getter(get_db_service()) as session:
        # Query if the record exists by x_consumer_username
        existing_tenant = (
            session.query(BotTenant)
            .filter(BotTenant.alias_id == x_consumer_username)
            .first()
        )
        if not existing_tenant:
            raise TenantNotFoundExc(
                f"Cannot find tenant id:{x_consumer_username} information"
            )


async def _validate_app_auth(app_id: str, sp: Span) -> dict:
    app_detail = await APPAuth().app_detail(app_id)
    sp.add_info_events({"kong-app-detail": json.dumps(app_detail, ensure_ascii=False)})
    if app_detail is None:
        raise AppAuthFailedExc(f"Cannot find appid:{app_id} authentication information")
    if app_detail.get("code") != 0:
        raise AppAuthFailedExc(app_detail.get("message", ""))
    if len(app_detail.get("data", [])) == 0:
        raise AppAuthFailedExc(f"Cannot find appid:{app_id} authentication information")
    return app_detail


async def _check_binding_status(
    auth_get_api_url: str, params: dict, timeout: aiohttp.ClientTimeout
) -> bool:
    """Check binding status, returns True if already bound"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            auth_get_api_url, params=params, timeout=timeout
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if response.status != 200:
                raise BotAuthFailedExc(
                    f"Failed to query binding status from third-party, "
                    f"code={response.status}"
                )
            if len(result.get("data", [])) > 0:
                return True

            return False


async def _perform_binding(
    auth_add_api_url: str,
    request_body: dict,
    headers: dict,
    timeout: aiohttp.ClientTimeout,
) -> None:
    """Perform binding operation"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            auth_add_api_url, json=request_body, headers=headers, timeout=timeout
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if response.status != 200:
                raise BotAuthFailedExc(
                    f"Failed to exec binding from third-party, "
                    f"code={response.status}"
                )

            if result.get("code") != 0:
                raise BotAuthFailedExc(result.get("message", "Binding failed"))

            return


@bot_manage_router.post("/bot")  # type: ignore[misc]
async def protocol_synchronization(
    x_consumer_username: Annotated[str, Header()], inputs: ProtocolSynchronization
) -> BotResponse:
    """Protocol Synchronization: Query data by ID, update if exists, create if not exists"""
    error: BotExc = BotExc(*c_0)
    resp_data: Union[dict] = dict()
    span = Span(
        app_id=x_consumer_username,
    )
    with span.start("ProtocolSynchronization") as sp:
        try:
            sp.add_info_events(
                {
                    "protocol-synchronization-inputs": inputs.model_dump_json(
                        by_alias=True
                    )
                }
            )

            await _validate_tenant(x_consumer_username)
            with session_getter(get_db_service()) as session:
                dsl_json = json.dumps(
                    inputs.dsl.model_dump(exclude_none=True), ensure_ascii=False
                )
                bot = (
                    session.query(Bot).filter(Bot.id == inputs.id).first()
                    if inputs.id
                    else None
                )
                current_time = datetime.now()
                if bot:
                    bot.dsl = dsl_json
                    bot.update_at = current_time
                else:
                    bot = Bot(
                        id=inputs.id,
                        app_id=x_consumer_username,
                        dsl=dsl_json,
                        create_at=current_time,
                        update_at=current_time,
                    )
                session.add(bot)
                session.commit()
                session.refresh(bot)
                resp_data["id"] = bot.id

        except BotExc as e:
            error = e
        except Exception as e:  # pylint: disable=broad-exception-caught
            error = BotProtocolSynchronizationFailedExc(str(e))

        return build_bot_response(error, resp_data)


@bot_manage_router.post("/bot/publish")  # type: ignore[misc]
async def publish(
    x_consumer_username: Annotated[str, Header()], inputs: Publish
) -> BotResponse:
    """Publish assistant: If dsl is provided, use it to create BotRelease; otherwise query from Bot table and create BotRelease"""
    error: BotExc = BotExc(*c_0)
    span = Span(
        app_id=x_consumer_username,
    )
    with span.start("Publish") as sp:
        try:
            sp.set_attribute("bot_id", inputs.bot_id)
            sp.set_attribute("version", inputs.version_name)
            sp.add_info_events(
                {"publish-inputs": inputs.model_dump_json(by_alias=True)}
            )

            await _validate_tenant(x_consumer_username)
            with session_getter(get_db_service()) as session:
                # Query Bot table data using bot_id
                bot = session.query(Bot).filter(Bot.id == inputs.bot_id).first()
                if not bot:
                    raise BotNotFoundExc(f"Bot with id {inputs.bot_id} not found")

                if bot.id != inputs.bot_id:
                    raise BotPublishFailedExc(f"Not allow")

                bot_version = (
                    session.query(BotRelease)
                    .filter(
                        BotRelease.bot_id == inputs.bot_id,
                        BotRelease.version == inputs.version_name,
                    )
                    .first()
                )
                if bot_version:
                    raise BotPublishDuplicatedExc(
                        f"Bot publish bot id:{inputs.bot_id} version:{inputs.version_name} duplicated"
                    )

                current_time = datetime.now()

                bot_release = BotRelease(
                    bot_id=inputs.bot_id,
                    version=inputs.version_name,
                    description=inputs.description,
                    dsl=bot.dsl,  # Use dsl from Bot table
                    create_at=current_time,
                    update_at=current_time,
                )
                session.add(bot_release)
                session.commit()
                session.refresh(bot_release)

                return build_bot_response(error, {"version_id": bot_release.id})
        except BotExc as e:
            error = e
        except Exception as e:  # pylint: disable=broad-exception-caught
            error = BotPublishFailedExc(str(e))

        return build_bot_response(error)


@bot_manage_router.post("/bot/auth")  # type: ignore[misc]
async def auth(
    x_consumer_username: Annotated[str, Header()], inputs: Auth
) -> BotResponse:
    """Bind capability: Query if already bound, perform binding operation if not bound"""
    error: BotExc = BotExc(*c_0)
    span = Span(
        app_id=x_consumer_username,
    )
    with span.start("Auth") as sp:
        try:
            # Get environment variables
            auth_get_api_url = os.getenv("AUTH_GET_API_URL")
            auth_add_api_url = os.getenv("AUTH_ADD_API_URL")
            auth_required_username = os.getenv("AUTH_REQUIRED_USERNAME")

            if (
                not auth_get_api_url
                or not auth_add_api_url
                or not auth_required_username
            ):
                raise BotAuthFailedExc("Required environment variables are not set")
            sp.set_attribute("version_id", inputs.version_id)
            sp.set_attribute("app_id", inputs.app_id)
            sp.add_info_events({"auth-inputs": inputs.model_dump_json(by_alias=True)})

            await _validate_tenant(x_consumer_username)
            # response format {'sid': 'app0001912f@hf19b24c8fdd65741502', 'code': 0, 'message': 'success', 'data': [{'appid': '', 'name': '', 'is_disable': False, 'auth_list': [{'api_key': '', 'api_secret': ''}], 'desc': '星辰租户'}]}{'sid': 'app0001912f@hf19b24c8fdd65741502', 'code': 0, 'message': 'success', 'data': [{'appid': '', 'name': '星辰租户', 'is_disable': False, 'auth_list': [{'api_key': '', 'api_secret': ''}], 'desc': '星辰租户'}]}
            app_detail = await _validate_app_auth(inputs.app_id, sp)
            print(app_detail)

            # Query app_detail firstly
            with session_getter(get_db_service()) as session:
                # Build query parameters
                params = {
                    "app_id": inputs.app_id,
                    "type": "agent",
                    "ability_id": str(inputs.version_id),
                }
                # Query if already bound
                timeout = aiohttp.ClientTimeout(total=5)
                try:
                    is_bound = await _check_binding_status(
                        auth_get_api_url, params, timeout
                    )
                    if is_bound:
                        raise BotAuthFailedExc(
                            "This capability id is already bound, no need to rebind"
                        )
                except aiohttp.ClientError as e:
                    raise BotAuthFailedExc(f"Failed to query binding status: {str(e)}")

                db_app_detail = (
                    session.query(AppAuthDetail)
                    .filter(
                        AppAuthDetail.release_id == inputs.version_id,
                        AppAuthDetail.app_id == inputs.app_id,
                    )
                    .first()
                )
                if db_app_detail:
                    raise BotAuthFailedExc(
                        "This capability id is already bound, no need to rebind"
                    )

                # If not bound, perform binding operation
                request_body = {
                    "app_id": inputs.app_id,
                    "type": "agent",
                    "ability_id": str(inputs.version_id),
                }

                headers = {
                    "x-consumer-username": auth_required_username,
                    "Content-Type": "application/json",
                }

                try:
                    await _perform_binding(
                        auth_add_api_url, request_body, headers, timeout
                    )
                except aiohttp.ClientError as e:
                    raise BotAuthFailedExc(f"Failed to bind: {str(e)}")

                api_key, api_secret = (
                    app_detail["data"][0]["auth_list"][0]["api_key"],
                    app_detail["data"][0]["auth_list"][0]["api_secret"],
                )
                db_app_detail = AppAuthDetail(
                    release_id=inputs.version_id,
                    app_id=inputs.app_id,
                    api_key=api_key,
                    api_secret=api_secret,
                )

                session.add(db_app_detail)
                session.commit()
                session.refresh(db_app_detail)

                return build_bot_response(error, None, "Binding succeeded")

        except BotExc as e:
            error = e
        except Exception as e:  # pylint: disable=broad-exception-caught
            error = BotAuthFailedExc(str(e))

        return build_bot_response(error)
