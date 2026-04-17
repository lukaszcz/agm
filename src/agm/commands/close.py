"""agm close."""

from __future__ import annotations

from agm.commands.args import CloseArgs
from agm.utils.project import branch_session_name, current_project_dir
from agm.utils.project_session import close_session


def run(args: CloseArgs) -> None:
    proj_dir = current_project_dir()
    session_name = branch_session_name(proj_dir, args.branch)
    close_session(branch=args.branch)
    print(f"Closed session {session_name}")
