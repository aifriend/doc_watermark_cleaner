import mimetypes
import os
import pickle
from pathlib import PurePath

import numpy as np


class ClassFile:

    def __init__(self):
        pass

    @staticmethod
    def get_file_root_name(file):
        file_name, file_ext = os.path.splitext(file)
        if not file_ext:
            return file_name
        return ClassFile.get_file_root_name(file_name)

    @staticmethod
    def has_file(path, file):
        try:
            file_list = list()
            if os.path.isdir(path):
                file_list = ClassFile.list_files(path)
            elif os.path.isfile(path):
                file_list.append(path)
            if file_list and file:
                for file_item in file_list:
                    if file in file_item:
                        return True
            return False
        except Exception as ex:
            raise Exception(ex)

    @staticmethod
    def list_files(path):
        """
        list all files under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_files_like(path, pattern):
        """
        list all files like a pattern under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if pattern in file:
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_pdf_files(path):
        """
        list all pdf files under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                file_root, file_extension = os.path.splitext(file)
                if 'pdf' in file_extension.lower():
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_files_ext(path, ext):
        """
        list all files under path with given extension
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                file_root, file_extension = os.path.splitext(file)
                if ext.lower() in file_extension.lower():
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def filter_files_root(source):
        file_filter = list()
        if isinstance(source, list):
            for file in source:
                file_root, file_extension = os.path.splitext(file)
                file_filter.append(file_root)
        elif isinstance(source, str) and os.path.isfile(source):
            file_root, file_extension = os.path.splitext(source)
            file_filter.append(file_root)
        return file_filter

    @staticmethod
    def filter_by_size(source):
        """
        filter all files with size greater than 0
        """
        file_filter = list()
        if isinstance(source, list):
            for file in source:
                if os.path.isfile(file) and os.path.getsize(file) > 0:
                    file_filter.append(file)
        elif isinstance(source, str):
            if os.path.isfile(source) and os.path.getsize(source) > 0:
                file_filter.append(source)
        return file_filter

    @staticmethod
    def filter_by_ext(source, ext):
        """
        filter all files with ext
        """
        file_filter = list()
        if isinstance(source, list):
            for file in source:
                file_name, file_ext = os.path.splitext(file)
                if ext.lower() in file_ext.lower():
                    file_filter.append(file)
        elif isinstance(source, str):
            file_name, file_ext = os.path.splitext(source)
            if ext.lower() in file_ext.lower():
                file_filter.append(source)
        return file_filter

    @staticmethod
    def filter_gram_duplicate(path, file_list):
        """
        filter all files with gram
        """
        filter_gram_list = ClassFile.filter_files_root(ClassFile.list_files_ext(path, "gram"))
        filter_list = file_list.copy()
        for file in file_list:
            file_root, _ = os.path.splitext(file)
            file_root, _ = os.path.splitext(file_root)
            if file_root in filter_gram_list:
                filter_list.remove(file)

        return filter_list

    @staticmethod
    def filter_duplicate(path, file_list, ext, level=1, filter_ext="PDF"):
        """
        filter all files with duplicates. Find file with EXT at index LEVEL
        and then compare with FILTER_EXT
        """
        filter_by_ext = ClassFile.list_files_ext(path, ext)
        while level > 0:
            filter_by_ext = ClassFile.filter_files_root(filter_by_ext)
            level -= 1
        filter_list = file_list.copy()
        for file_ext in filter_by_ext:
            file_cmp = f"{file_ext}.{filter_ext}"
            if file_cmp in file_list:
                filter_list.remove(file_cmp)

        return filter_list

    @staticmethod
    def get_file_ext(file):
        """
        get the extension of a file
        """
        return os.path.splitext(file)[1]

    @staticmethod
    def get_dir_name(file):
        """
        get the the full path dir container of the file
        """
        return os.path.dirname(file)

    @staticmethod
    def get_containing_dir_name(file):
        """
        get just the name of the containing dir of the file
        """
        return PurePath(file).parent.name

    @staticmethod
    def file_base_name(file_name):
        """
        get the path and name of the file without the extension
        """
        if '.' in file_name:
            separator_index = file_name.index('.')
            base_name = file_name[:separator_index]
            return base_name
        else:
            return file_name

    @staticmethod
    def get_file_name(path):
        """
        get the name of the file without the extension
        """
        file_name = os.path.basename(path)
        return ClassFile.file_base_name(file_name)

    @staticmethod
    def create_dir(directory):
        """
        create a directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def list_to_file(set_, file_):
        """
        save a list to a file using pickle dump
        """
        with open(file_, 'wb') as fp:
            pickle.dump(sorted(list(set_)), fp)

    @staticmethod
    def to_txtfile_by_line(data: list, file_, mode="w"):
        """
        save data as a text file
        """
        with open(file_, mode) as output:
            output.writelines(data)

    @staticmethod
    def to_txtfile(data: str, file_, mode="w"):
        """
        save data as a text file
        """
        with open(file_, mode) as output:
            output.write(data)

    @staticmethod
    def file_to_list(file_, binary=True, encoding="ISO-8859-1"):
        """
        read a list from a pickle file
        """
        list_ = []
        if os.path.getsize(file_) > 0:
            if binary:
                with open(file_, 'rb') as fp:
                    list_ = pickle.load(fp)
            else:
                with open(file_, 'r', encoding=encoding) as fp:
                    while True:
                        data = fp.readline()
                        if not data:
                            break
                        list_.append(data.splitlines())
        return sorted(list(list_))

    @staticmethod
    def csv_to_numpy_image(csv_file):
        """
        load numpy image from csv file
        """
        np.loadtxt(csv_file)

    @staticmethod
    def get_text(filename, encoding="ISO-8859-1"):
        """
        read from file as text
        """
        with open(filename, 'r', encoding=encoding) as f:
            file_content = f.read()
        return file_content

    @staticmethod
    def save_sparse_csr(filename, mat):
        """
        save a sparse vector as a list of its elements
        """
        _ = mat.toarray().tofile(filename, sep=",", format="%f")

    @staticmethod
    def save_model(filename, model):
        """
        save scikit-learn model
        """
        pickle.dump(model, open(filename, 'wb'))

    @staticmethod
    def load_model(filename):
        """
        load scikit-learn model
        """
        try:
            if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                with open(filename, 'rb') as fp:
                    return pickle.load(fp)
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def has_text_file(path, depth=1):
        try:
            file_test_list = list()
            if os.path.isdir(path):
                file_test_list = ClassFile.list_files_ext(path, "txt")
            elif os.path.isfile(path):
                file_test_list.append(path)
            for i, file_test in enumerate(file_test_list):
                if i > depth:
                    break
                mime = mimetypes.guess_type(file_test)
                if mime and mime[0] == "text/plain":
                    return True
            return False
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def has_gram_file(path, depth=1):
        try:
            file_test_list = list()
            if os.path.isdir(path):
                file_test_list = ClassFile.list_files_ext(path, "gram")
            elif os.path.isfile(path):
                file_test_list.append(path)
            for i, file_test in enumerate(file_test_list):
                if i > depth:
                    break
                if "gram" in file_test.lower():
                    return True
            return False
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def file_to_list_unsorted(file_):
        """
        read a unsorted list from a pickle file
        :param file_: file full path
        :return: list with elements
        """
        with open(file_, 'rb') as fp:
            list_ = pickle.load(fp)
        return list(list_)

    @staticmethod
    def list_to_file_unsorted(list_, file_):
        """
        save a list to a file using pickle dump
        :param list_: list to save
        :param file_: destination file full path
        :return:
        """
        with open(file_, 'wb') as fp:
            pickle.dump(list(list_), fp)
