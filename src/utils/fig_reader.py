"""
Date: 2023-11-14
Authors: Juan Pablo Triana and Kuniaki Iwanami
Date: 2023-11-14
"""
import os
import pandas

class CXReader:
    """
    Class that would perform the following:
    1.) Read all images from the data folder.
    2.) Read the dataframes with the metadata of the files.
    """

    def __init__(self, folder_path:str, df_name:str):
        """
        Parameters:
            folder_path(str): String path that would access de data
            dataframe_name(str): String name of the dataframe to access
            metadata from.
        """
        # Define the folder_path
        self.folder_path = folder_path
        self.df_name = df_name
        pass

    