"""File tools - Manage files and notebooks in Databricks workspace.

Consolidated into 1 tool:
- manage_workspace_files: list, get_info, upload, download, delete
"""

from typing import Any, Dict, Optional

from databricks_tools_core.file import (
    delete_from_workspace as _delete_from_workspace,
    download_from_workspace as _download_from_workspace,
    get_workspace_object_info as _get_workspace_object_info,
    list_workspace_objects as _list_workspace_objects,
    upload_to_workspace as _upload_to_workspace,
)

from ..server import mcp


@mcp.tool(timeout=300)
def manage_workspace_files(
    action: str,
    workspace_path: str,
    # For list:
    max_results: int = 100,
    object_type_filter: Optional[str] = None,
    name_contains: Optional[str] = None,
    notebooks_modified_after: Optional[int] = None,
    # For upload:
    local_path: Optional[str] = None,
    max_workers: int = 10,
    overwrite: bool = True,
    # For download/delete:
    local_destination: Optional[str] = None,
    export_format: str = "SOURCE",
    recursive: bool = False,
) -> Dict[str, Any]:
    """Manage workspace files and notebooks: list, get_info, upload, download, delete.

    Actions:
    - list: List workspace objects. Supports recursive traversal, object_type_filter,
      name_contains, notebooks_modified_after, and max_results.
      Returns: {objects: [{path, object_type, language, size, created_at, modified_at, object_id}],
      returned_count, truncated}.
    - get_info: Get metadata for a workspace object. Requires workspace_path.
      Returns: {path, object_type, language, size, created_at, modified_at, object_id}.
    - upload: Upload files/folders to workspace. Requires local_path, workspace_path.
      Supports files, folders, globs, tilde expansion.
      max_workers: Parallel upload threads (default 10). overwrite: Replace existing (default True).
      Returns: {local_folder, remote_folder, total_files, successful, failed, success, failed_uploads}.
    - download: Export a workspace file/notebook/directory to a local path. Requires
      local_destination. export_format defaults to SOURCE. Directories require recursive=True.
      Returns: {workspace_path, local_path, export_format, success, error}.
    - delete: Delete file/folder from workspace. Requires workspace_path.
      recursive=True for non-empty folders. Has safety checks for protected paths.
      Returns: {workspace_path, success, error}.

    workspace_path format: /Workspace/Users/user@example.com/path/to/files"""
    act = action.lower()

    if act == "list":
        result = _list_workspace_objects(
            workspace_path=workspace_path,
            recursive=recursive,
            object_type_filter=object_type_filter,
            name_contains=name_contains,
            notebooks_modified_after=notebooks_modified_after,
            max_results=max_results,
        )
        return {
            "objects": [obj.to_dict() for obj in result.objects],
            "returned_count": result.returned_count,
            "truncated": result.truncated,
            "error": result.error,
        }

    elif act == "get_info":
        result = _get_workspace_object_info(workspace_path=workspace_path)
        if result.error:
            return {"workspace_path": workspace_path, "error": result.error}
        return result.to_dict()

    elif act == "upload":
        if not local_path:
            return {"error": "upload requires: local_path"}
        result = _upload_to_workspace(
            local_path=local_path,
            workspace_path=workspace_path,
            max_workers=max_workers,
            overwrite=overwrite,
        )
        return {
            "local_folder": result.local_folder,
            "remote_folder": result.remote_folder,
            "total_files": result.total_files,
            "successful": result.successful,
            "failed": result.failed,
            "success": result.success,
            "failed_uploads": [
                {"local_path": r.local_path, "error": r.error} for r in result.get_failed_uploads()
            ]
            if result.failed > 0
            else [],
        }

    elif act == "download":
        if not local_destination:
            return {"error": "download requires: local_destination"}
        result = _download_from_workspace(
            workspace_path=workspace_path,
            local_destination=local_destination,
            export_format=export_format,
            overwrite=overwrite,
            recursive=recursive,
        )
        return {
            "workspace_path": result.workspace_path,
            "local_path": result.local_path,
            "export_format": result.export_format,
            "success": result.success,
            "error": result.error,
        }

    elif act == "delete":
        result = _delete_from_workspace(
            workspace_path=workspace_path,
            recursive=recursive,
        )
        return {
            "workspace_path": result.workspace_path,
            "success": result.success,
            "error": result.error,
        }

    else:
        return {"error": f"Invalid action '{action}'. Valid actions: list, get_info, upload, download, delete"}
