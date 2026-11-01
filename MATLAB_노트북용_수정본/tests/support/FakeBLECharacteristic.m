classdef FakeBLECharacteristic < handle
    properties
        DataAvailableFcn=[]
        Packet=uint8([])
    end
    methods
        function p=read(obj,varargin)
            p=obj.Packet;
        end
        function emit(obj,seq,ms)
            body=zeros(1,45,'uint8');
            body(1:4)=typecast(uint32(ms),'uint8');
            for k=1:6
                body(5+(k-1)*2:6+(k-1)*2)=typecast(uint16(k*100),'uint8');
            end
            for k=1:7
                body(17+(k-1)*4:20+(k-1)*4)=typecast(single(k/10),'uint8');
            end
            body(45)=1;
            for k=1:3
                obj.Packet=[uint8(159+k) typecast(uint32(seq),'uint8') body((k-1)*15+1:k*15)];
                if ~isempty(obj.DataAvailableFcn), obj.DataAvailableFcn(obj,[]); end
            end
        end
    end
end
