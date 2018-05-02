# helper.py
# store numerous helper functions


import itertools as it
# import mmap
# import sys
import os

# -----------------------------------------------------------
# User convience functions
# -----------------------------------------------------------

# Should add helper functions that can read the .npz files directly
# possibly translate them to text or csv for quick analzying


# -----------------------------------------------------------
# MEMORY MAPPED HELPER FUNCTIONS
# -----------------------------------------------------------

def readlines(mmFile, number_of_lines):
    """if we need to skip mulitple lines"""
    for x in range(0, number_of_lines):
        mmFile.readline()


def find_string_in_file(mmFile, filePath, targetString):
    """wrapper that raises error if no substr can be found
    finds the last occurance of a substring in memory mapped file
    """
    location = mmFile.find(targetString.encode(encoding="utf-8"))

    if location == -1:
        # couldn't find target string in file
        s = "It seems \"{:s}\" was not present in the file\n" \
          + "\"{:s}\"\nCheck that the previous calculation didn't fail"
        raise Exception(s.format(targetString, filePath))
    return location


def rfind_string_in_file(mmFile, filePath, targetString):
    """wrapper that raises error if no substr can be found
    finds the last occurance of a substring in memory mapped file"""
    location = mmFile.rfind(targetString.encode(encoding="utf-8"))

    if location == -1:
        # couldn't find target string in file
        s = "It seems \"{:s}\" was not present in the file\n" \
          + "\"{:s}\"\nCheck that the previous calculation didn't fail"
        raise Exception(s.format(targetString, filePath))
    return location


def skip_back_lines(mm, numLines, startIndex):
    """gives the byte location numLines lines before
    the given byte location startIndex"""
    '''Factored out to simplify handling of n and offset'''
    for placeHolder in it.repeat(None, numLines):
        tempstartIndex = mm.rfind(b'\n', 0, startIndex)
        if tempstartIndex < 0:
            break
        startIndex = tempstartIndex

    return startIndex


def skip_forward_lines(mm, numLines, startIndex):
    """gives the byte location numLines lines after
    the given byte location startIndex"""
    '''Factored out to simplify handling of n and offset'''
    for placeHolder in it.repeat(None, numLines):
        tempstartIndex = mm.find(b'\n', startIndex + 1)
        if tempstartIndex == -1:
            break
        startIndex = tempstartIndex

    return startIndex

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------


def verify_file_exists(filePath):
    """x"""
    if not os.path.isfile(filePath):
        s = "The provided path {:s} does not appear to be a file, or it does not exist"
        raise FileNotFoundError(s.format(filePath))


# -----------------------------------------------------------
