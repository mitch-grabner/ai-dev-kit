"""Tests for manage_workspace_files MCP wrapper logic."""

from unittest.mock import patch

from databricks_mcp_server.tools.file import manage_workspace_files
from databricks_tools_core.file import (
    DeleteResult,
    WorkspaceDownloadResult,
    WorkspaceListResult,
    WorkspaceObjectInfo,
)


def _manage_workspace_files(**kwargs):
    return manage_workspace_files.__wrapped__(**kwargs)


def test_manage_workspace_files_list():
    result_obj = WorkspaceListResult(
        objects=[
            WorkspaceObjectInfo(
                path="/Workspace/Users/test@example.com/notebook",
                object_type="NOTEBOOK",
                language="PYTHON",
                size=123,
                created_at=1,
                modified_at=2,
                object_id=3,
            )
        ],
        returned_count=1,
        truncated=False,
    )
    with patch("databricks_mcp_server.tools.file._list_workspace_objects", return_value=result_obj) as mock_list:
        result = _manage_workspace_files(
            action="list",
            workspace_path="/Workspace/Users/test@example.com",
            recursive=True,
            object_type_filter="NOTEBOOK",
            name_contains="note",
            notebooks_modified_after=10,
            max_results=25,
        )

    mock_list.assert_called_once_with(
        workspace_path="/Workspace/Users/test@example.com",
        recursive=True,
        object_type_filter="NOTEBOOK",
        name_contains="note",
        notebooks_modified_after=10,
        max_results=25,
    )
    assert result["returned_count"] == 1
    assert result["truncated"] is False
    assert result["objects"][0]["object_type"] == "NOTEBOOK"


def test_manage_workspace_files_get_info():
    info = WorkspaceObjectInfo(
        path="/Workspace/Users/test@example.com/file.py",
        object_type="FILE",
        size=99,
    )
    with patch("databricks_mcp_server.tools.file._get_workspace_object_info", return_value=info):
        result = _manage_workspace_files(
            action="get_info",
            workspace_path="/Workspace/Users/test@example.com/file.py",
        )

    assert result["path"] == "/Workspace/Users/test@example.com/file.py"
    assert result["object_type"] == "FILE"
    assert result["size"] == 99


def test_manage_workspace_files_get_info_error():
    info = WorkspaceObjectInfo(
        path="/Workspace/Users/test@example.com/missing",
        error="not found",
    )
    with patch("databricks_mcp_server.tools.file._get_workspace_object_info", return_value=info):
        result = _manage_workspace_files(
            action="get_info",
            workspace_path="/Workspace/Users/test@example.com/missing",
        )

    assert result["workspace_path"] == "/Workspace/Users/test@example.com/missing"
    assert result["error"] == "not found"


def test_manage_workspace_files_download():
    download_result = WorkspaceDownloadResult(
        workspace_path="/Workspace/Users/test@example.com/notebook",
        local_path="/tmp/notebook.py",
        export_format="SOURCE",
        success=True,
    )
    with patch("databricks_mcp_server.tools.file._download_from_workspace", return_value=download_result) as mock_download:
        result = _manage_workspace_files(
            action="download",
            workspace_path="/Workspace/Users/test@example.com/notebook",
            local_destination="/tmp",
            export_format="SOURCE",
            overwrite=False,
            recursive=True,
        )

    mock_download.assert_called_once_with(
        workspace_path="/Workspace/Users/test@example.com/notebook",
        local_destination="/tmp",
        export_format="SOURCE",
        overwrite=False,
        recursive=True,
    )
    assert result["success"] is True
    assert result["local_path"] == "/tmp/notebook.py"


def test_manage_workspace_files_download_requires_local_destination():
    result = _manage_workspace_files(
        action="download",
        workspace_path="/Workspace/Users/test@example.com/notebook",
    )

    assert result["error"] == "download requires: local_destination"


def test_manage_workspace_files_delete_still_works():
    delete_result = DeleteResult(
        workspace_path="/Workspace/Users/test@example.com/file.py",
        success=True,
    )
    with patch("databricks_mcp_server.tools.file._delete_from_workspace", return_value=delete_result):
        result = _manage_workspace_files(
            action="delete",
            workspace_path="/Workspace/Users/test@example.com/file.py",
        )

    assert result["success"] is True


def test_manage_workspace_files_invalid_action_lists_valid_actions():
    result = _manage_workspace_files(
        action="bogus",
        workspace_path="/Workspace/Users/test@example.com/file.py",
    )

    assert "list, get_info, upload, download, delete" in result["error"]
