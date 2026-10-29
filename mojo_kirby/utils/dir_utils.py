import os
import pathlib
import shutil
from collections import defaultdict
import logging

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich.tree import Tree


def find_files_by_extension(folder_path, extension):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(root, file)


def make_directory(path, prompt_if_exists=False):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info("New directory [u]{}[/u] created.".format(path))
    else:
        logging.info("Directory [u]{}[/u] already exists.".format(path))
        if not prompt_if_exists:
            return

        rm_dir = Confirm.ask(
            "Do you want to delete the folder? "
            "[bold red blink]This will permanently remove its contents.[/bold red blink]",
            default=False,
        )

        if rm_dir:
            shutil.rmtree(path)
            os.makedirs(path)
            logging.info("New directory [u]{}[/u] created.".format(path))
        else:
            # kill process
            logging.info("Killing process.")
            exit()

