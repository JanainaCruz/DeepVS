from os import stat, listdir, path
from stat import ST_MODE, S_ISREG, S_ISDIR

#>------------------------------------------------------------------------------
def traverseDirectory(fileName, callback, extension=None, verbose=False):
    """
    Recursively traverses a directory and its subdirectories.
    For each visited file the "callback" funtion is called, having the file name 
    as a parameter. 
     
    @param string fileName  : a file name or directory name
    @param function callback: a python function
    @param string extension : file extension (e.g. 'xml', 'txt').
    """
    mode = stat(fileName)[ST_MODE]
    if S_ISDIR(mode):
        files = listdir(fileName)
        files.sort()
        for f in files:
            pathname = path.join(fileName, f)
            mode = stat(pathname)[ST_MODE]
            if S_ISDIR(mode):
                # It's a directory, recurse into it
                traverseDirectory(pathname, callback, extension)
            elif S_ISREG(mode) and (not extension or \
                pathname.endswith(extension)):
                # It's a good file, call the callback function
                if verbose:
                    print 'Processing %s...' % pathname
                callback(pathname)
            else:
                # Unknown file type, print a message
                if verbose:
                    print 'Skipping %s' % pathname
    elif S_ISREG(mode):
        if verbose:
            print 'Processing %s...' % pathname
        callback(fileName)
    else:
        print 'Invalid file: %s' % pathname
