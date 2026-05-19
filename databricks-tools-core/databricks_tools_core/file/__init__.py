"""
File - Workspace File Operations

Functions for managing files and folders in Databricks Workspace.

Note: For Unity Catalog Volume file operations, use the unity_catalog module.
"""

from .workspace import (
    DeleteResult,
    FolderUploadResult,
    WorkspaceDownloadResult,
    WorkspaceListResult,
    WorkspaceObjectInfo,
    UploadResult,
    delete_from_workspace,
    download_from_workspace,
    get_workspace_object_info,
    list_workspace_objects,
    upload_to_workspace,
)

__all__ = [
    # Workspace file operations
    "DeleteResult",
    "FolderUploadResult",
    "WorkspaceDownloadResult",
    "WorkspaceListResult",
    "WorkspaceObjectInfo",
    "UploadResult",
    "delete_from_workspace",
    "download_from_workspace",
    "get_workspace_object_info",
    "list_workspace_objects",
    "upload_to_workspace",
]
