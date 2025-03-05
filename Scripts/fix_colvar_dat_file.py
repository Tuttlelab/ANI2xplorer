#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:03:51 2025

@author: bwb16179
"""

import os


def update_file(filepath):
    """
    Script to re-write the top line of the *COLVAR*.dat file (output of US)
    The file is read in as a pandas DataFrame but a column named 'bb.bias' can't
    be indexed and must be redesignated as just bias

    filepath: direct filepath to the specific COVLAR file for a specific window

    """

    # attempt to catch all variations (accounting for differing rc's)
    pattern = [
        "#! FIELDS time dR1 dR2 dR3 dR4 rc bb.bias",
        "#! FIELDS time dAB dAC rc bb.bias",
        "#! FIELDS time dAB dAC dAD rc bb.bias",
        "#! FIELDS time dAB dAC rc bias",
    ]

    try:
        with open(filepath, "r") as file:
            lines = file.readlines()

        # Check if the first line matches the specified pattern
        if lines and lines[0].strip() in pattern:
            lines[0] = "#! FIELDS time dAB dAC rc bias\n"
            with open(filepath, "w") as file:
                file.writelines(lines)
            print(f"Updated file: {filepath}")
        else:
            print(f"Skipped file: {filepath} (no matching first line)")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")


def search_and_update(folder):
    for root, _, files in os.walk(folder):
        # print(root, _, files)
        for filename in files:
            if filename.startswith("COLVAR_d_") and filename.endswith(".dat"):
                filepath = os.path.join(root, filename)
                print(filepath)
                update_file(filepath)


folder_to_search = "Working_Folder/ANI/Reaction_1C/18"
search_and_update(folder_to_search)
