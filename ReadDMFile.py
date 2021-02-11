#!/usr/bin/python
"""Python module for reading DM3 files"""

import os
import time
import struct

#import tictoc


### binary data reading functions ###

def readLong(f):
    """Read 4 bytes as integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('>l', read_bytes)[0]

def readLLong(f, version): 
        if version == 3:
            return readLong(f)
        else:  
            read_bytes = f.read(8)
            return struct.unpack('>q', read_bytes)[0]
        
def readInt(f):
    """Read 2 bytes as integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('>h', read_bytes)[0]

def readShort(f):
    """Read 2 bytes as integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('>h', read_bytes)[0]


def readByte(f):
    """Read 1 byte as integer in file f"""
    read_bytes = f.read(1)
    return struct.unpack('>b', read_bytes)[0]


def readString(f, len_=1):
    """Read len_ bytes as a string in file f"""
    read_bytes = f.read(len_)
    str_fmt = '>'+str(len_)+'s'
    return struct.unpack( str_fmt, read_bytes )[0]

def readBool(f):
    """Read 1 byte as boolean in file f"""
    read_val = readByte(f)
    return (read_val!=0)

def readChar(f):
    """Read 1 byte as char in file f"""
    read_bytes = f.read(1)
    return struct.unpack('c', read_bytes)[0]


def readLEShort(f):
    """Read 2 bytes as *little endian* integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('<h', read_bytes)[0]

def readLELong(f):
    """Read 4 bytes as *little endian* integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<l', read_bytes)[0]

def readLEUShort(f):
    """Read 2 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('<H', read_bytes)[0]

def readLEULong(f):
    """Read 4 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<L', read_bytes)[0]

def readLEFloat(f):
    """Read 4 bytes as *little endian* float in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<f', read_bytes)[0]

def readLEDouble(f):
    """Read 8 bytes as *little endian* double in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<d', read_bytes)[0]

def readLongLong(f):
    """Read 8 bytes as *little endian* int64 in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<q', read_bytes)[0]

def readULLong(f):
    """Read 8 bytes as *little endian* int64 in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<Q', read_bytes)[0]

## constants for encoded data types ##
SHORT = 2
LONG = 3
USHORT = 4
ULONG = 5
FLOAT = 6
DOUBLE = 7
BOOLEAN = 8
CHAR = 9
OCTET = 10
LLONG = 11
ULLONG = 12
STRUCT = 15
STRING = 18
ARRAY = 20

readFunc = {
    SHORT: readLEShort,
    LONG: readLELong,
    USHORT: readLEUShort,
    ULONG: readLEULong,
    FLOAT: readLEFloat,
    DOUBLE: readLEDouble,
    BOOLEAN: readBool,
    CHAR: readChar,
    OCTET: readChar,    
    LLONG: readLongLong,
    ULLONG: readULLong
}

celltags=['ImageList 2 ImageData Calibrations Dimension 1 Scale',
    'ImageList 2 ImageData Calibrations Dimension 1 Units',
    'ImageList 2 ImageData Dimensions 1',
    'ImageList 2 ImageData Dimensions 2',
    'ImageList 2 ImageData Dimensions 3',
    'ImageList 2 ImageData Data']

        
class DM3(object):
    """DM3 object. """
    def _readTagGroup(self):
        
        # is the group sorted?
        sorted_ = readByte(self._f)
        isSorted = (sorted_ == 1)
        # is the group open?
        opened = readByte(self._f)
        isOpen = (opened == 1)
        # number of Tags
        nTags = readLLong(self._f, self._fileVersion)
        # read Tags
        for i in xrange( nTags ):
            self._readTagEntry(i)
        return 1
    
    def _readTagEntry(self, index):
        # is data or a new group?
        isdata = readByte(self._f)
        
        self._level = self._level + 1
        # get tag label if exists
        lenTagLabel = readShort(self._f)
        if ( lenTagLabel != 0 ):
            tagLabel = readString(self._f, lenTagLabel)
        else:
            tagLabel = str(index+1)
     #   print "tagLabel = ", tagLabel
        
        self._tags[self._level] = tagLabel
                
        if (self._fileVersion == 4):
            totalBytes = readLong(self._f)
        
        if isdata == 21:
            self._readTagType()
        elif isdata == 20:
            self._readTagGroup()  # increments curGroupLevel
        else:
            raise Exception("Unknown TagEntry type")
            
        self._tags[self._level] = ""
        self._level = self._level-1
        return 1  
    
    def _readTagType(self):
        dum = readLong(self._f)
        if (dum != 623191333):
            raise Exception("Illegal TagType value")
            
        deflen = readLLong(self._f, self._fileVersion)
        EncType = readLLong(self._f, self._fileVersion)
        result = self._readData(EncType)
        
        index = self._checkTags()
        if (index >0):
            self._output[index] = result                
        return 1                 
        
    def _checkTags(self):
        r = -1
        for i in range(len(celltags)):
            ok = 1
            c = celltags[i].split()            
            j = 0            
            while (ok and j < len(c)):
                ok = (c[j] == '*') or (c[j] == self._tags[j])
                j = j+1
            if ok:
                return i
        return r
    
    def _readData(self, ftype, num=1):
      #  print "ftype =", ftype
        width = self._encodedTypeSize(ftype)
        
        if(width > 0):
            x = [1] * num
            for i in xrange(num):
                x[i] = self._readNativeData(ftype, width)
        elif(ftype == 15):
            x = self._readStruct()
        elif(ftype == 18):
            length = readLong(self._f)
            x = self._readStringData(length) 
        elif(ftype == 20):
            x = self._readArray()
        else:
            x = -1                                 
        return x
    
    def _encodedTypeSize(self, eT):
        # returns the size in bytes of the data type
        if eT == 0:
            width = 0
        elif eT in (BOOLEAN, CHAR, OCTET):
            width = 1
        elif eT in (SHORT, USHORT):
            width = 2
        elif eT in (LONG, ULONG, FLOAT):
            width = 4
        elif eT in (DOUBLE, LLONG, ULLONG):
            width = 8
        else:
            # returns -1 for unrecognised types
            width = -1
        return width
               
    def _readNativeData(self, encodedType, etSize):
        # reads ordinary data types
        if encodedType in readFunc:
            val = readFunc[encodedType](self._f)
        else:
            raise Exception("rND, " + hex(self._f.tell()) 
                            + ": Unknown data type " + str(encodedType))
        return val
            
    def _readStruct(self):
        StructNameLength=readLLong(self._f, self._fileVersion)
        NumFields=readLLong(self._f, self._fileVersion)
        x=[-1] *  NumFields
        FieldNameLength = [' '] * NumFields
        FieldType = [-1] * NumFields
        for i in xrange(NumFields):
            FieldNameLength[i]=readLLong(self._f, self._fileVersion)
            FieldType[i]=readLLong(self._f, self._fileVersion)
        StructName= self._readStringData(StructNameLength)
        for i in xrange(NumFields):
            FieldNameLen=FieldNameLength[i]
            FieldName=self._readStringData(FieldNameLength[i])
            x[i] =self._readData(FieldType[i])                    
        return x
        
    def _readStringData(self, stringSize):
        # reads string data
        if ( stringSize <= 0 ):
            rString = ""
        else:
            ## !!! *Unicode* string (UTF-16)... convert to Python unicode str
            rString = readString(self._f, stringSize)
            rString = unicode(rString, "utf_16_le")
        return rString

    def _readArray(self):
        ArrayType = readLLong(self._f,self._fileVersion)
        
        if ( ArrayType == 15 ):
            StructNameLength=readLLong(self._f, self._fileVersion)
            NumFields=readLLong(self._f, self._fileVersion)
            x=[-1] *  NumFields
            FieldNameLength = [' '] * NumFields
            FieldType = [-1] * NumFields
            for i in xrange(NumFields):
                FieldNameLength[i]=readLLong(self._f, self._fileVersion)
                FieldType[i]=readLLong(self._f, self._fileVersion)
        
        ArrayLength = readLLong(self._f, self._fileVersion)
        
        if ( ArrayType == 15 ):
            for j in xrange(ArrayLength):
                for i in xrange(NumFields):
                    FieldNameLen=FieldNameLength[i]
                    FieldName=self._readStringData(FieldNameLength[i])
                    x[i] =self._readData(FieldType[i])                    
            return x            
        elif (ArrayType == 4):
            x = [[]] * ArrayLength
            for j in xrange(ArrayLength):
                x[j] = self._readData(ArrayType)        
            return x
        elif(ArrayLength > 1000):
            x = self._readData(ArrayType, ArrayLength)  
            return x
        else:
            x = [[]] * ArrayLength
            for j in xrange(ArrayLength):
                x[j] = self._readData(ArrayType)        
            return x
                    
    
    def __init__(self, filename):
        """DM3 object: parses DM3 file."""
        # - track currently read group
        self._level = -1
        self._tags = [""] * 100
        self._output = [[]] * 6
        self._filename = filename
        # - open file for reading
        self._f = open( self._filename, 'rb' )
        # - create Tags repositories
        isDM3 = True
        ## read header (first 3 4-byte int)
        # get version
        self._fileVersion = readLong(self._f)
        if ( self._fileVersion < 3 or self._fileVersion > 4 ):
            isDM3 = False
        # get indicated file size
        fileSize = readLong(self._f)
        # get byte-ordering
        lE = readLong(self._f)
        littleEndian = (lE == 1)
        if not littleEndian:
            isDM3 = False
        # check file header, raise Exception if not DM3
        if not isDM3:
            raise Exception("%s does not appear to be a DM3 file." 
                            % os.path.split(self._filename)[1]) 
        
        # ... then read it
        self._readTagGroup()                           
    
        self._sx= self._output[0]
        self._units = self._output[1]
        self._xdim=self._output[2]
        self._ydim=self._output[3]
        self._zdim=self._output[4]         # no. of images in a stack
        
        if len(self._zdim)<1 or any(self._zdim)<1:
            self._zdim=1

        self._m=self._output[5]                                                                
                
    @property
    def filename(self):
        """Returns full file path."""
        return self._filename

    @property
    def imagedata(self):
        return self._m
    
    @property
    def imagexdim(self):
        return self._xdim
     
    @property
    def imageydim(self):
        return self._ydim        
    @property
    def imagezdim(self):
        return self._zdim                   