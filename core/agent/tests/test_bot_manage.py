"""Test bot management API endpoints in bot_manage module"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from common.otlp import sid as sid_module

from agent.api.schemas.bot_manage_inputs import Auth, ProtocolSynchronization, Publish
from agent.api.v2.bot_manage import (
    _check_binding_status,
    _perform_binding,
    _validate_app_auth,
    _validate_tenant,
    auth,
    protocol_synchronization,
    publish,
)
from agent.domain.models.bot import Bot, BotRelease, BotTenant
from agent.exceptions.bot_exc import BotExc


@dataclass
class _DummySidGen:
    """Simple sid generator for testing environment."""

    value: str = "test-sid"

    def gen(self) -> str:  # pragma: no cover - only for testing environment
        return self.value


@pytest.fixture(autouse=True)
def _setup_test_environment() -> None:
    """Automatically inject environment fixes for all tests.

    - Ensure `sid_generator2` is initialized to avoid `Span` construction failure.
    """
    # Initialize sid generator to avoid Span throwing "sid_generator2 is not initialized"
    if sid_module.sid_generator2 is None:
        sid_module.sid_generator2 = _DummySidGen()  # type: ignore[assignment]


class TestValidateTenant:
    """Test _validate_tenant function"""

    @pytest.mark.asyncio
    async def test_validate_tenant_success(self) -> None:
        """Test successful tenant validation"""
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_username",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                mock_session = MagicMock()
                mock_query = MagicMock()
                mock_filter = MagicMock()
                mock_query.filter.return_value = mock_filter
                mock_filter.first.return_value = mock_tenant
                mock_session.query.return_value = mock_query
                mock_session_getter.return_value.__enter__.return_value = mock_session
                mock_session_getter.return_value.__exit__.return_value = False

                await _validate_tenant("test_username")

                mock_session.query.assert_called_once_with(BotTenant)
                mock_query.filter.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_tenant_not_found(self) -> None:
        """Test tenant validation when tenant is not found"""

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                mock_session = MagicMock()
                mock_query = MagicMock()
                mock_filter = MagicMock()
                mock_query.filter.return_value = mock_filter
                mock_filter.first.return_value = None
                mock_session.query.return_value = mock_query
                mock_session_getter.return_value.__enter__.return_value = mock_session
                mock_session_getter.return_value.__exit__.return_value = False

                with pytest.raises(BotExc) as exc_info:
                    await _validate_tenant("test_username")
                assert exc_info.value.c == 40604


class TestCheckBindingStatus:
    """Test _check_binding_status function"""

    @pytest.mark.asyncio
    async def test_check_binding_status_bound(self) -> None:
        """Test checking binding status when already bound"""
        mock_response_data = {"code": 0, "data": [{"id": "1"}]}
        timeout = aiohttp.ClientTimeout(total=5)

        # Mock HTTP response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.raise_for_status = MagicMock()

        # Mock ClientSession.get() to return an async context manager
        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_context.__aexit__ = AsyncMock(return_value=False)

        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _check_binding_status("http://test.com", {}, timeout)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_binding_status_not_bound(self) -> None:
        """Test checking binding status when not bound"""
        mock_response_data = {"code": 0, "data": []}
        timeout = aiohttp.ClientTimeout(total=5)

        # Mock HTTP response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.raise_for_status = MagicMock()

        # Mock ClientSession.get() to return an async context manager
        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_context.__aexit__ = AsyncMock(return_value=False)

        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _check_binding_status("http://test.com", {}, timeout)
            assert not result

    @pytest.mark.asyncio
    async def test_check_binding_status_code_not_zero(self) -> None:
        """Test checking binding status when response code is not 0"""
        mock_response_data = {"code": 1, "message": "error", "data": []}
        timeout = aiohttp.ClientTimeout(total=5)

        # Mock HTTP response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.raise_for_status = MagicMock()

        # Mock ClientSession.get() to return an async context manager
        mock_get_context = AsyncMock()
        mock_get_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_context.__aexit__ = AsyncMock(return_value=False)

        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_get_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _check_binding_status("http://test.com", {}, timeout)
            # When code is not 0 but data is empty, should return False
            assert result is False


class TestPerformBinding:
    """Test _perform_binding function"""

    @pytest.mark.asyncio
    async def test_perform_binding_success(self) -> None:
        """Test successful binding operation"""
        mock_response_data = {"code": 0, "message": "success"}
        timeout = aiohttp.ClientTimeout(total=5)

        # Mock HTTP response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.raise_for_status = MagicMock()

        # Mock ClientSession.post() to return an async context manager
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_context.__aexit__ = AsyncMock(return_value=False)

        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await _perform_binding("http://test.com", {}, {}, timeout)

    @pytest.mark.asyncio
    async def test_perform_binding_failure(self) -> None:
        """Test binding operation failure"""
        mock_response_data = {"code": 1, "message": "binding failed"}
        timeout = aiohttp.ClientTimeout(total=5)

        # Mock HTTP response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)
        mock_resp.raise_for_status = MagicMock()

        # Mock ClientSession.post() to return an async context manager
        mock_post_context = AsyncMock()
        mock_post_context.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_context.__aexit__ = AsyncMock(return_value=False)

        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_context)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(BotExc) as exc_info:
                await _perform_binding("http://test.com", {}, {}, timeout)
            assert exc_info.value.c == 40602


class TestProtocolSynchronization:
    """Test protocol_synchronization endpoint"""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create mock database session"""
        session = MagicMock()
        # Setup query chain mock
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        session.query.return_value = mock_query

        session.add = MagicMock()
        session.commit = MagicMock()

        # Mock refresh to update object attributes
        def mock_refresh(obj: Any) -> None:
            """Mock refresh that sets object attributes"""
            if isinstance(obj, Bot):
                if not hasattr(obj, "id") or not obj.id:
                    obj.id = "test_bot_id"
                if not hasattr(obj, "create_at") or not obj.create_at:
                    obj.create_at = datetime.now()
                if not hasattr(obj, "update_at") or not obj.update_at:
                    obj.update_at = datetime.now()

        session.refresh = MagicMock(side_effect=mock_refresh)
        return session

    @pytest.fixture
    def mock_tenant_session(self) -> MagicMock:
        """Create mock database session for tenant validation"""
        session = MagicMock()
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_app",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        # Setup query chain mock for tenant
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_tenant
        session.query.return_value = mock_query
        return session

    @pytest.mark.asyncio
    async def test_protocol_synchronization_create_new_bot(
        self, mock_session: MagicMock, mock_tenant_session: MagicMock
    ) -> None:
        """Test creating a new bot via protocol synchronization"""
        from agent.api.schemas.bot_manage_inputs import (
            DslInputs,
            ModelInputs,
            ModelPropertiesInputs,
        )

        dsl = DslInputs(
            name="test_bot",
            model=ModelInputs(
                name="test_model",
                type="test_type",
                properties=ModelPropertiesInputs(
                    id="model_id", url="http://test.com", token="token"
                ),
            ),
        )
        inputs = ProtocolSynchronization(id="test_bot_id", dsl=dsl)

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                # First call for tenant validation, second for main logic
                mock_session_getter.return_value.__enter__.side_effect = [
                    mock_tenant_session,
                    mock_session,
                ]
                mock_session_getter.return_value.__exit__.return_value = False

                response = await protocol_synchronization("test_app", inputs)

                assert response.code == 0
                assert response.data is not None
                assert response.data["id"] == "test_bot_id"
                assert response.data["app_id"] == "test_app"
                assert response.data["pub_status"] == 0
                mock_session.add.assert_called_once()
                mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_protocol_synchronization_update_existing_bot(
        self, mock_session: MagicMock, mock_tenant_session: MagicMock
    ) -> None:
        """Test updating an existing bot via protocol synchronization"""
        from agent.api.schemas.bot_manage_inputs import (
            DslInputs,
            ModelInputs,
            ModelPropertiesInputs,
        )

        dsl = DslInputs(
            name="test_bot",
            model=ModelInputs(
                name="test_model",
                type="test_type",
                properties=ModelPropertiesInputs(
                    id="model_id", url="http://test.com", token="token"
                ),
            ),
        )
        inputs = ProtocolSynchronization(id="test_bot_id", dsl=dsl)

        # Mock existing bot
        existing_bot = Bot(
            id="test_bot_id",
            app_id="old_app",
            dsl="old_dsl",
            pub_status=0,
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        # Update mock to return existing bot
        mock_session.query.return_value.filter.return_value.first.return_value = (
            existing_bot
        )

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                # First call for tenant validation, second for main logic
                mock_session_getter.return_value.__enter__.side_effect = [
                    mock_tenant_session,
                    mock_session,
                ]
                mock_session_getter.return_value.__exit__.return_value = False

                response = await protocol_synchronization("test_app", inputs)

                assert response.code == 0
                assert response.data is not None
                assert response.data["id"] == "test_bot_id"
                assert existing_bot.app_id == "test_app"
                assert existing_bot.dsl is not None  # Should be updated with new dsl
                mock_session.add.assert_called_once_with(existing_bot)
                mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_protocol_synchronization_auth_failed(self) -> None:
        """Test protocol synchronization when tenant validation fails"""
        from agent.api.schemas.bot_manage_inputs import (
            DslInputs,
            ModelInputs,
            ModelPropertiesInputs,
        )

        dsl = DslInputs(
            name="test_bot",
            model=ModelInputs(
                name="test_model",
                type="test_type",
                properties=ModelPropertiesInputs(
                    id="model_id", url="http://test.com", token="token"
                ),
            ),
        )
        inputs = ProtocolSynchronization(id="test_bot_id", dsl=dsl)

        mock_tenant_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None  # Tenant not found
        mock_tenant_session.query.return_value = mock_query

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                mock_session_getter.return_value.__enter__.return_value = (
                    mock_tenant_session
                )
                mock_session_getter.return_value.__exit__.return_value = False

                response = await protocol_synchronization("test_app", inputs)

                assert response.code != 0
                assert response.data is None

    @pytest.mark.asyncio
    async def test_protocol_synchronization_exception(self) -> None:
        """Test protocol synchronization when exception occurs"""
        from agent.api.schemas.bot_manage_inputs import (
            DslInputs,
            ModelInputs,
            ModelPropertiesInputs,
        )

        dsl = DslInputs(
            name="test_bot",
            model=ModelInputs(
                name="test_model",
                type="test_type",
                properties=ModelPropertiesInputs(
                    id="model_id", url="http://test.com", token="token"
                ),
            ),
        )
        inputs = ProtocolSynchronization(id="test_bot_id", dsl=dsl)

        mock_tenant_session = MagicMock()
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_app",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_tenant
        mock_tenant_session.query.return_value = mock_query

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                # First call succeeds for tenant, second call raises exception
                mock_session_getter.return_value.__enter__.side_effect = [
                    mock_tenant_session,
                    Exception("Database error"),
                ]

                response = await protocol_synchronization("test_app", inputs)

                assert response.code != 0
                assert response.data is None


class TestPublish:
    """Test publish endpoint"""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create mock database session"""
        session = MagicMock()
        # Setup query chain mock
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None
        session.query.return_value = mock_query

        session.add = MagicMock()
        session.commit = MagicMock()

        # Mock refresh to update object attributes
        def mock_refresh(obj: Any) -> None:
            """Mock refresh that sets object attributes"""
            if isinstance(obj, BotRelease):
                obj.id = "release_id"
            elif isinstance(obj, Bot):
                if not hasattr(obj, "id") or not obj.id:
                    obj.id = "test_bot_id"

        session.refresh = MagicMock(side_effect=mock_refresh)
        return session

    @pytest.fixture
    def mock_tenant_session(self) -> MagicMock:
        """Create mock database session for tenant validation"""
        session = MagicMock()
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_app",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        # Setup query chain mock for tenant
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_tenant
        session.query.return_value = mock_query
        return session

    @pytest.mark.asyncio
    async def test_publish_bot_not_found(
        self, mock_session: MagicMock, mock_tenant_session: MagicMock
    ) -> None:
        """Test publish when bot is not found"""
        inputs = Publish(
            bot_id="test_bot_id", version="v1.0", description="test", dsl=None
        )

        # Mock bot not found
        mock_session.query.return_value.filter.return_value.first.return_value = None

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                # First call for tenant validation, second for main logic
                mock_session_getter.return_value.__enter__.side_effect = [
                    mock_tenant_session,
                    mock_session,
                ]
                mock_session_getter.return_value.__exit__.return_value = False

                response = await publish("test_app", inputs)

                assert response.code != 0
                assert response.data is None

    @pytest.mark.asyncio
    async def test_publish_auth_failed(self) -> None:
        """Test publish when tenant validation fails"""
        inputs = Publish(
            bot_id="test_bot_id", version="v1.0", description="test", dsl=None
        )

        mock_tenant_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None  # Tenant not found
        mock_tenant_session.query.return_value = mock_query

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                mock_session_getter.return_value.__enter__.return_value = (
                    mock_tenant_session
                )
                mock_session_getter.return_value.__exit__.return_value = False

                response = await publish("test_app", inputs)

                assert response.code != 0
                assert response.data is None

    @pytest.mark.asyncio
    async def test_publish_exception(self) -> None:
        """Test publish when exception occurs"""
        inputs = Publish(
            bot_id="test_bot_id", version="v1.0", description="test", dsl=None
        )

        mock_tenant_session = MagicMock()
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_app",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_tenant
        mock_tenant_session.query.return_value = mock_query

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                # First call succeeds for tenant, second call raises exception
                mock_session_getter.return_value.__enter__.side_effect = [
                    mock_tenant_session,
                    Exception("Database error"),
                ]

                response = await publish("test_app", inputs)

                assert response.code != 0
                assert response.data is None

    @pytest.mark.asyncio
    async def test_publish_version_duplicated(
        self, mock_session: MagicMock, mock_tenant_session: MagicMock
    ) -> None:
        """Test publish when version is already duplicated"""
        inputs = Publish(
            bot_id="test_bot_id", version="v1.0", description="test", dsl=None
        )

        # Mock existing bot
        existing_bot = Bot(
            id="test_bot_id",
            app_id="test_app",
            dsl='{"name": "test"}',
            pub_status=0,
            create_at=datetime.now(),
            update_at=datetime.now(),
        )

        # Mock existing BotRelease (version duplicated)
        existing_release = BotRelease(
            id="existing_release_id",
            bot_id="test_bot_id",
            version="v1.0",
            description="existing",
            dsl='{"name": "existing"}',
            create_at=datetime.now(),
            update_at=datetime.now(),
        )

        # Create a fresh mock session
        test_mock_session = MagicMock()
        test_mock_session.add = MagicMock()
        test_mock_session.commit = MagicMock()
        test_mock_session.refresh = MagicMock()

        # Setup query chain for Bot query
        mock_bot_query = MagicMock()
        mock_bot_filter = MagicMock()
        mock_bot_query.filter.return_value = mock_bot_filter
        mock_bot_filter.first.return_value = existing_bot

        # Setup query chain for BotRelease query (should return existing_release for duplicated version)
        mock_release_query = MagicMock()
        mock_release_filter = MagicMock()
        mock_release_query.filter = lambda *args, **kwargs: mock_release_filter
        mock_release_filter.first = lambda: existing_release

        # Make session.query return different queries based on model class type
        def query_func(model_class: Any) -> Any:
            if model_class is Bot:
                return mock_bot_query
            elif model_class is BotRelease:
                return mock_release_query
            # Default
            default_query = MagicMock()
            default_filter = MagicMock()
            default_query.filter = lambda *args, **kwargs: default_filter
            default_filter.first.return_value = None
            return default_query

        test_mock_session.query = query_func

        with patch("agent.api.v2.bot_manage.get_db_service"):
            with patch("agent.api.v2.bot_manage.session_getter") as mock_session_getter:
                mock_session_getter.return_value.__enter__.side_effect = [
                    mock_tenant_session,
                    test_mock_session,
                ]
                mock_session_getter.return_value.__exit__.return_value = False

                response = await publish("test_app", inputs)

                # Should return error code for duplicated version
                assert response.code == 40605  # BotPublishDuplicatedExc code
                assert response.data is None


class TestValidateAppAuth:
    """Test _validate_app_auth function"""

    @pytest.mark.asyncio
    async def test_validate_app_auth_success(self) -> None:
        """Test successful app auth validation"""
        mock_span = MagicMock()
        mock_app_detail = {"code": 0, "data": [{"id": "app1"}]}

        with patch("agent.api.v2.bot_manage.APPAuth") as mock_app_auth_class:
            mock_app_auth = MagicMock()
            mock_app_auth.app_detail = AsyncMock(return_value=mock_app_detail)
            mock_app_auth_class.return_value = mock_app_auth

            await _validate_app_auth("test_app_id", mock_span)

            mock_app_auth.app_detail.assert_called_once_with("test_app_id")
            mock_span.add_info_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_app_auth_none(self) -> None:
        """Test app auth validation when app_detail is None"""
        mock_span = MagicMock()

        with patch("agent.api.v2.bot_manage.APPAuth") as mock_app_auth_class:
            mock_app_auth = MagicMock()
            mock_app_auth.app_detail = AsyncMock(return_value=None)
            mock_app_auth_class.return_value = mock_app_auth

            with pytest.raises(BotExc) as exc_info:
                await _validate_app_auth("test_app_id", mock_span)
            assert exc_info.value.c == 40040  # AppAuthFailedExc code

    @pytest.mark.asyncio
    async def test_validate_app_auth_code_not_zero(self) -> None:
        """Test app auth validation when code is not 0"""
        mock_span = MagicMock()
        mock_app_detail = {"code": 1, "message": "Auth failed"}

        with patch("agent.api.v2.bot_manage.APPAuth") as mock_app_auth_class:
            mock_app_auth = MagicMock()
            mock_app_auth.app_detail = AsyncMock(return_value=mock_app_detail)
            mock_app_auth_class.return_value = mock_app_auth

            with pytest.raises(BotExc) as exc_info:
                await _validate_app_auth("test_app_id", mock_span)
            assert exc_info.value.c == 40040  # AppAuthFailedExc code

    @pytest.mark.asyncio
    async def test_validate_app_auth_empty_data(self) -> None:
        """Test app auth validation when data is empty"""
        mock_span = MagicMock()
        mock_app_detail = {"code": 0, "data": []}

        with patch("agent.api.v2.bot_manage.APPAuth") as mock_app_auth_class:
            mock_app_auth = MagicMock()
            mock_app_auth.app_detail = AsyncMock(return_value=mock_app_detail)
            mock_app_auth_class.return_value = mock_app_auth

            with pytest.raises(BotExc) as exc_info:
                await _validate_app_auth("test_app_id", mock_span)
            assert exc_info.value.c == 40040  # AppAuthFailedExc code


class TestAuth:
    """Test auth endpoint"""

    @pytest.fixture
    def mock_tenant_session(self) -> MagicMock:
        """Create mock database session for tenant validation"""
        session = MagicMock()
        mock_tenant = BotTenant(
            id="test_tenant_id",
            name="test_tenant",
            alias_id="test_app",
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        # Setup query chain mock for tenant
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = mock_tenant
        session.query.return_value = mock_query
        return session

    @pytest.mark.asyncio
    async def test_auth_already_bound(self, mock_tenant_session: MagicMock) -> None:
        """Test auth when capability is already bound"""
        inputs = Auth(version_id=123, app_id="test_app")

        with patch.dict(
            os.environ,
            {
                "AUTH_GET_API_URL": "http://test.com/get",
                "AUTH_ADD_API_URL": "http://test.com/add",
                "AUTH_REQUIRED_USERNAME": "test_user",
            },
        ):
            with patch("agent.api.v2.bot_manage.get_db_service"):
                with patch(
                    "agent.api.v2.bot_manage.session_getter"
                ) as mock_session_getter:
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_tenant_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    mock_response_data = {"code": 0, "data": [{"id": "1"}]}

                    # Mock HTTP response
                    mock_resp = AsyncMock()
                    mock_resp.status = 200
                    mock_resp.json = AsyncMock(return_value=mock_response_data)
                    mock_resp.raise_for_status = MagicMock()

                    # Mock ClientSession.get() to return an async context manager
                    mock_get_context = AsyncMock()
                    mock_get_context.__aenter__ = AsyncMock(return_value=mock_resp)
                    mock_get_context.__aexit__ = AsyncMock(return_value=False)

                    # Mock ClientSession
                    mock_session = AsyncMock()
                    mock_session.get = MagicMock(return_value=mock_get_context)
                    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session.__aexit__ = AsyncMock(return_value=False)

                    with patch("aiohttp.ClientSession", return_value=mock_session):
                        response = await auth("test_app", inputs)

                        assert response.code == 0
                        assert (
                            response.message
                            == "This capability id is already bound, no need to rebind"
                        )

    @pytest.mark.asyncio
    async def test_auth_binding_success(self, mock_tenant_session: MagicMock) -> None:
        """Test successful binding operation"""
        inputs = Auth(version_id=123, app_id="test_app")

        with patch.dict(
            os.environ,
            {
                "AUTH_GET_API_URL": "http://test.com/get",
                "AUTH_ADD_API_URL": "http://test.com/add",
                "AUTH_REQUIRED_USERNAME": "test_user",
            },
        ):
            with patch("agent.api.v2.bot_manage.get_db_service"):
                with patch(
                    "agent.api.v2.bot_manage.session_getter"
                ) as mock_session_getter:
                    # First call for tenant validation
                    mock_session_getter.return_value.__enter__.side_effect = [
                        mock_tenant_session,
                    ]
                    mock_session_getter.return_value.__exit__.return_value = False
                with patch(
                    "agent.api.v2.bot_manage._validate_app_auth"
                ) as mock_validate_app_auth:
                    mock_validate_app_auth.return_value = None

                    # Mock GET response (not bound)
                    mock_get_response_data = {"code": 0, "data": []}
                    mock_get_resp = AsyncMock()
                    mock_get_resp.status = 200
                    mock_get_resp.json = AsyncMock(return_value=mock_get_response_data)
                    mock_get_resp.raise_for_status = MagicMock()

                    # Mock POST response (binding success)
                    mock_post_response_data = {"code": 0, "message": "success"}
                    mock_post_resp = AsyncMock()
                    mock_post_resp.status = 200
                    mock_post_resp.json = AsyncMock(
                        return_value=mock_post_response_data
                    )
                    mock_post_resp.raise_for_status = MagicMock()

                    # Mock ClientSession.get() and post() to return async context managers
                    mock_get_context = AsyncMock()
                    mock_get_context.__aenter__ = AsyncMock(return_value=mock_get_resp)
                    mock_get_context.__aexit__ = AsyncMock(return_value=False)

                    mock_post_context = AsyncMock()
                    mock_post_context.__aenter__ = AsyncMock(
                        return_value=mock_post_resp
                    )
                    mock_post_context.__aexit__ = AsyncMock(return_value=False)

                    # Mock ClientSession
                    mock_session = AsyncMock()
                    mock_session.get = MagicMock(return_value=mock_get_context)
                    mock_session.post = MagicMock(return_value=mock_post_context)
                    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session.__aexit__ = AsyncMock(return_value=False)

                    with patch("aiohttp.ClientSession", return_value=mock_session):
                        response = await auth("test_app", inputs)

                        assert response.code == 0
                        assert response.message == "Binding succeeded"

    @pytest.mark.asyncio
    async def test_auth_missing_env_vars(self) -> None:
        """Test auth when environment variables are missing"""
        inputs = Auth(version_id=123, app_id="test_app")

        with patch.dict(os.environ, {}, clear=True):
            response = await auth("test_app", inputs)

            assert response.code != 0

    @pytest.mark.asyncio
    async def test_auth_check_binding_failed(
        self, mock_tenant_session: MagicMock
    ) -> None:
        """Test auth when checking binding status fails"""
        inputs = Auth(version_id=123, app_id="test_app")

        with patch.dict(
            os.environ,
            {
                "AUTH_GET_API_URL": "http://test.com/get",
                "AUTH_ADD_API_URL": "http://test.com/add",
                "AUTH_REQUIRED_USERNAME": "test_user",
            },
        ):
            with patch("agent.api.v2.bot_manage.get_db_service"):
                with patch(
                    "agent.api.v2.bot_manage.session_getter"
                ) as mock_session_getter:
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_tenant_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    # Mock ClientSession that raises error on get
                    mock_session = AsyncMock()
                    mock_session.get = AsyncMock(
                        side_effect=aiohttp.ClientError("Connection error")
                    )
                    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session.__aexit__ = AsyncMock(return_value=False)

                    with patch("aiohttp.ClientSession", return_value=mock_session):
                        response = await auth("test_app", inputs)

                        assert response.code != 0

    @pytest.mark.asyncio
    async def test_auth_binding_failed(self, mock_tenant_session: MagicMock) -> None:
        """Test auth when binding operation fails"""
        inputs = Auth(version_id=123, app_id="test_app")

        with patch.dict(
            os.environ,
            {
                "AUTH_GET_API_URL": "http://test.com/get",
                "AUTH_ADD_API_URL": "http://test.com/add",
                "AUTH_REQUIRED_USERNAME": "test_user",
            },
        ):
            with patch("agent.api.v2.bot_manage.get_db_service"):
                with patch(
                    "agent.api.v2.bot_manage.session_getter"
                ) as mock_session_getter:
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_tenant_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    # Mock GET response (not bound)
                    mock_get_response_data = {"code": 0, "data": []}
                    mock_get_resp = AsyncMock()
                    mock_get_resp.status = 200
                    mock_get_resp.json = AsyncMock(return_value=mock_get_response_data)
                    mock_get_resp.raise_for_status = MagicMock()

                    # Mock ClientSession.get() to return an async context manager
                    mock_get_context = AsyncMock()
                    mock_get_context.__aenter__ = AsyncMock(return_value=mock_get_resp)
                    mock_get_context.__aexit__ = AsyncMock(return_value=False)

                    # Mock ClientSession that raises error on post
                    mock_session = AsyncMock()
                    mock_session.get = MagicMock(return_value=mock_get_context)
                    mock_session.post = AsyncMock(
                        side_effect=aiohttp.ClientError("Binding error")
                    )
                    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_session.__aexit__ = AsyncMock(return_value=False)

                    with patch("aiohttp.ClientSession", return_value=mock_session):
                        response = await auth("test_app", inputs)

                        assert response.code != 0

    @pytest.mark.asyncio
    async def test_auth_app_auth_failed(self) -> None:
        """Test auth when tenant validation fails"""
        inputs = Auth(version_id=123, app_id="test_app")

        mock_tenant_session = MagicMock()
        mock_query = MagicMock()
        mock_filter = MagicMock()
        mock_query.filter.return_value = mock_filter
        mock_filter.first.return_value = None  # Tenant not found
        mock_tenant_session.query.return_value = mock_query

        with patch.dict(
            os.environ,
            {
                "AUTH_GET_API_URL": "http://test.com/get",
                "AUTH_ADD_API_URL": "http://test.com/add",
                "AUTH_REQUIRED_USERNAME": "test_user",
            },
        ):
            with patch("agent.api.v2.bot_manage.get_db_service"):
                with patch(
                    "agent.api.v2.bot_manage.session_getter"
                ) as mock_session_getter:
                    mock_session_getter.return_value.__enter__.return_value = (
                        mock_tenant_session
                    )
                    mock_session_getter.return_value.__exit__.return_value = False

                    response = await auth("test_app", inputs)

                    assert response.code != 0

    @pytest.mark.asyncio
    async def test_auth_exception(self) -> None:
        """Test auth when exception occurs"""
        inputs = Auth(version_id=123, app_id="test_app")

        with patch.dict(
            os.environ,
            {
                "AUTH_GET_API_URL": "http://test.com/get",
                "AUTH_ADD_API_URL": "http://test.com/add",
                "AUTH_REQUIRED_USERNAME": "test_user",
            },
        ):
            with patch("agent.api.v2.bot_manage.get_db_service"):
                with patch(
                    "agent.api.v2.bot_manage.session_getter"
                ) as mock_session_getter:
                    # Raise exception during tenant validation
                    mock_session_getter.return_value.__enter__.side_effect = Exception(
                        "Unexpected error"
                    )

                    response = await auth("test_app", inputs)

                    assert response.code != 0
