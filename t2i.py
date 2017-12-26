import json
import pickle

class T2I(object):

    def __init__(self,idict=None,with_padding="__PADDING__",with_unknown="__UNK__"):
        """
        `idict`: can be dictionary, string ending with ".json" or ".pkl/.pickle", or None
        `with_padding`: if not None, this string will be entry index 0
        `with_unknown`: if not None, this string will be entry index 1
        """

        self.padding=None
        self.unknown=None
        if isinstance(idict,str):
            if idict.endswith(".json"):
                with open(idict,"rt") as f:
                    self.idict,self.padding,self.unknown=json.load(f)
            elif idict.endswith(".pkl") or idict.endswith(".pickle"):
                with open(idict,"rb") as f:
                    self.idict,self.padding,self.unknown=pickle.load(f)
        elif isinstance(idict,dict):
            self.idict=idict
            self.padding=self.idict.get(with_padding)
            self.unknown=self.idict.get(with_unknown)
        elif idict is None:
            self.idict={}
            if with_padding is not None:
                self.padding=0
                self.idict[with_padding]=self.padding
            if with_unknown is not None:
                self.unknown=1
                self.idict[with_unknown]=self.unknown


    def save(self,name):
        """
        Save the dictionary
        `name`: string with file name, can end with .json or .pickle/.pkl
        """
        if name.endswith(".json"):
            with open(name,"wt") as f:
                json.dump((self.idict,self.padding,self.unknown),f)
        elif name.endswith(".pickle") or name.endswith(".pkl"):
            with open(name,"wb") as f:
                pickle.dump((self.idict,self.padding,self.unknown),f)
        else:
            raise ValueError("File type cannot be guessed from extension. Supported are .json .pkl .pickle.: "+self.idict)

    def __call__(self,inp,string_as_sequence=False,train=True):
        """
        Turn input to indices. If train, new entries are inserted into the dict, otherwise the unknown entry is used.
        `inp`: by default, a string is translated into single index, a sequence is translated into a list of indices.
        `string_as_sequence`: if True, tread strings as sequences, effectively producing character level indices
        """
        if isinstance(inp,str) and (not string_as_sequence or len(inp)==1):
            if train:
                return self.idict.setdefault(inp,len(self.idict))
            else:
                return self.idict.get(inp,self.unknown)
        else:
            return list(self(item,string_as_sequence,train) for item in inp)

        
if __name__=="__main__":
    t2i=T2I()
    print("string")
    print("hi",t2i("hi"))
    print("hi there hi",t2i("hi there hi".split()))
    print(t2i.idict)
    
    t2i_char=T2I()
    print()
    print("character")
    print("hi",t2i_char("hi",string_as_sequence=True))
    print("hi there hi",t2i_char("hi there hi".split(),string_as_sequence=True))
    print(t2i_char.idict)
    
          
    
