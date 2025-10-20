import numpy as np
import struct
import os

class IGBWriter:
    """
    Class IGBWriter
    Implements a writer in IGB format
    """


    def __init__(self, config: dict = None):
        self._fname   : str        = 'out.igb'
        self._nt      : int        = None
        self._nx      : int        = None
        self._ny      : int        = None
        self._nz      : int        = None
        self._org_t   : float      = 0.0
        self._Tend    : float      = None
        self._dim_t   : float      = None
        self._units_x : str        = None
        self._units_y : str        = None
        self._units_z : str        = None
        self._units_t : str        = None
        self._units   : str        = None
        self.__fobj                = None
        self.__hwrt   : bool       = True
        self.__data   : np.ndarray = None         # used for manual operations

        if(config is not None):
            for attribute in self.__dict__.keys():
                if attribute[1:] in config.keys():
                    setattr(self, attribute, config[attribute[1:]])

        if ( (self._dim_t is None) and (self._Tend is not None) ):
            self._dim_t = self._Tend-self._org_t
        elif( (self._Tend is None) and (self._dim_t is not None)):
            self._Tend = self._org_t + self._dim_t

    def set_fname(self,fname: str):
        """ set_fname(fname) sets the output file name
        """
        self._fname = fname

    def set_space_units(self,space_units: str):
        """ set_space_units(space_units) sets the spaces units (x,y, and z) to space_units (string)
        """
        self._units_x = space_units
        self._units_y = space_units
        self._units_z = space_units

    def set_time_unit(self,time_unit: str):
        """ set_time_unit(space_unit) sets the time unit to time_unit (string)
        """
        self._units_t = time_unit

    def set_variable_unit(self,variable_unit:str):
        """ set_variable_unit(variable_unit) sets the variable unit unit to variable_unit (string)
        """
        self._units = variable_unit    
    
    def set_nt(self,nt: int):
        """ set_nt(nt) assigns the nb of time steps
        """
        self._nt = nt

    def set_nx(self,nx: int):
        """ set_nx(nx) assigns the space dimension of the problem
        """
        self._nx = nx

    def set_ny(self,ny: int = 1):
        """ set_ny(ny) assigns the y space dimension of the problem
        """
        self._ny = ny

    def set_nz(self,nz:int =1):
        """ set_nz(nz) assigns the z space dimension of the problem
        """
        self._nz = nz

    def set_org_t(self,t0:float):
        """ set_org_t(t0): sets the initial time (org_t) to t0
        """
        self._org_t = t0
        if self._Tend is not None:
            self._dim_t = self._Tend-self._org_t
        elif self._dim_t is not None:
            self._Tend = self._org_t + self._dim_t
             
    def set_dim_t(self,dim_t: float):
        """ set_dim_t(dim_t) sets the temporal dimension (converts from time step to actual time)
        """
        self._dim_t = dim_t
        self._Tend  = self._org_t + self._dim_t

    def set_data(self,data:np.ndarray):
        """set_data(data): sets the internal variable of the data to the data data-array
        """
        self.__data = data
        
    def nx(self) -> int:
        """ nx(): returns the space dimension
        """
        return(self._nx)                  

    def ny(self) -> int:
        """ ny(): returns y the space dimension
        """
        return(self._ny)

    def nz(self) -> int:
        """ nz(): returns z the space dimension
        """
        return(self._nz)

    def nt(self) -> int:
        """ nt(): returns the number of time steps
        """
        return(self._nt)                  

    def org_t(self) -> float:
        """ Tstart(): returns the time origin """
        return (self._org_t)

    def dim_t(self) -> float:
        """ dim_t(): returns the factor to convert from time step to time units
        """
        return(self._dim_t)

    def units_x(self) -> str:
        """units_x() returns the x unit
        """
        return(self._units_x)
        
    def units_y(self) -> str:
        """units_y() returns the y unit
        """
        return(self._units_y)

    def units_z(self) -> str:
        """units_z() returns the z unit
        """
        return(self._units_z)

    def units_t(self) -> str:
        """units_t() returns the temporal unit
        """
        return(self._units_t)

    def units(self) -> str:
        """units() returns the unknown unit
        """
        return(self._units)

    def data(self) -> np.ndarray:
        """data(): returns the ndarray with stored data
        the array is allocated OLNY with data/IGBReader initialisation
        """
        return(self.__data)

    def imshow(self,data: np.ndarray):
        """ imshow(data): writes the data to the file
        """
        if self.__hwrt:
            self.__write_header()  
        for ix in range(self._nx):
            self.__fobj.write(struct.pack('f', data[ix]))

    def wait(self):
        '''wait function; usually for plot on screen '''
        self.__fobj.close()
        for x in [0,1,2]:
            pass

    def write_data_to_file(self):
        ''' This function is to write all the data to a file; to use igberiter as a standalone'''
        if self.__hwrt:
            self.__write_header()  
        for n in range(self._nt):
            self.__fobj.write(struct.pack('={}f'.format(self.__data.shape[1]),*(self.__data[n,:])))
        self.__fobj.close()        


    def initialise_from_data(self, header:dict, data: np.ndarray):
        """initialise_from_im tdata(header,data) initialises an IGBWriter object passing an header and a numpy ndarray of data
        """ 
        self.__parse_header(header)
        self.__data = data

    def initialise_from_IGBReader(self, reader):
        """initialise_from_IGBReader(reader) initialises an IGBWriter object from an IGBreader object
        """ 
        self.__parse_header(reader.header())
        self.__data = reader.data()


    def __del__(self):
        '''destructor: closes the file '''
        self.__fobj.close()

    def __write_header(self):
        '''writes the header to the file.'''
        self.___openfile()
        if self._dim_t is  None:
            self._dim_t = self._nt-1
        header ='x:{} y:{} z:{} t:{} type:float systeme:little_endian org_t:{} dim_t:{}'.format(self._nx, self._ny, self._nz, self._nt, self._org_t,self._dim_t)
        
        if (self._units_x is not None) and (self._units_y is not None) and (self._units_z is not None):
            header ='{} unites_x:{} unites_y:{} unites_z:{}'.format(header,self._units_x,self._units_y,self._units_z)
        
        if self._units_t is not None:
            header ='{} unites_t:{}'.format(header,self._units_t)

        if self._units is not None:
            header ='{} unites:{}'.format(header,self._units)

        for isp in range(1022-len(header)):
            header = header+' '
        header = header+'\n\f'
        header = ''.join('\n' if i % 80 == 0 else char for i, char in enumerate(header, 1))
        self.__fobj.write(header.encode('utf8'))
        self.__hwrt = False

    def ___openfile(self):
        '''openfile(): checks that the output directory exists and
        that the file name has the extension .igb; opens a file object'''
        try:
            if self.__fobj is None:
                fdir  = os.path.split(self._fname)[0]
                fname = os.path.split(self._fname)[-1]
                tmp   = fname.rfind('.')
                if tmp ==-1:
                    self._fname = '{}.igb'.format(self._fname)
                if fdir and (not os.path.exists(fdir)):
                    os.makedirs(fdir)
                self.__fobj = open(self._fname, 'wb')            
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __parse_header(self,header:dict):
        '''__parse_header(header) used to copy from an igbreader object; it parses the header
        '''

        for attribute, map_attribute in zip(['_nx','_ny','_nz','_nt'], ['x','y','z','t']):
            if map_attribute in header.keys():
                setattr(self, attribute, header[map_attribute])
        
        
        for attribute in self.__dict__.keys():    
            if attribute[1:] in header.keys():
                setattr(self, attribute, header[attribute[1:]])

        if ( (self._dim_t is None) and (self._Tend is not None) ):
            self._dim_t = self._Tend-self._org_t
        elif( (self._Tend is None) and (self._dim_t is None)):
            self._Tend = self._org_t + self._dim_t
    

