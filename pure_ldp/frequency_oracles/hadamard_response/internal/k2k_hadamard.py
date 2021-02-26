# Copyright 2017 Department of Electrical and Computer Engineering, Cornell University. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This is a package for locally private data transmission. 

import random
from pure_ldp.frequency_oracles.hadamard_response.internal.functions import *

#the Hadamard randamized responce when \epsilon < 1
class Hadamard_Rand_high_priv:
    def __init__(self, absz, pri_para, encode_acc = 0): # absz: alphabet size, pri_para: privacy parameter
        #set encode_acc = 1 to enable fast encoding by intializing hadamard matrix
        self.insz = absz #input alphabet size k
        self.outsz = int(math.pow(2,math.ceil(math.log(absz+1,2)))) #output alphabet size: smallest exponent of 2 which is bigger than k
        self.outbit = int(math.ceil(math.log(absz+1,2))) #output bit length
        self.exp = math.exp(pri_para) #privacy parameter
        self.pri_para = 1/(1+math.exp(pri_para)) #flipping probability to maintain local privacy
        #self.permute, self.reverse = Random_Permutation(absz) #Initialize random permutation
        
        self.ifencode_acc = encode_acc #whether to accelerate encoding process
        if encode_acc == 1:
            self.H = Hadarmard_init(self.outsz) # initialize Hadarmard matrix
            
    def parity_check(self,x,y): #check if the hadamard entry is one (return 0 for if the entry is 1)
        z = x&y #bit and
        z_bit = bin(z)[2:].zfill(self.outbit)
        check = 0
        for i in range(0,self.outbit): #check = \sum_i (x_i y_i) （mod 2）
            check = check^int(z_bit[i]) 
        return check
                                  
    def encode_symbol(self,in_symbol):  # encode a single symbol into a privatized version
        bitin = bin(int(in_symbol)+1)[2:].zfill(self.outbit) #map the input symbol x to x+1 since we are not using the first column of the matrix
        out1 = random.randint(0,math.pow(2,self.outbit)-1) #get a random number in the output alphabet as one potential output
        bitout1 = bin(out1)[2:].zfill(self.outbit)
        for i in range(0,self.outbit): #flip the bit of out1 corresponding to the left most bit in (in_symbol+1) which is one to get the other potential output
            if int(bitin[i]) == 1:
                out2 = out1 ^ (pow(2,self.outbit - i -1))
                break   
        #bitout2 = bin(out2)[2:].zfill(self.outbit)
        
        if self.ifencode_acc == 1:
            check = 1 - self.H[int(in_symbol)+1][out1]
        else:
            check = 0
            for i in range(0,self.outbit): # check if the Hadamard entry at position (in_symbol+1, out1) is one or not
                check = check^(int(bitout1[i])&int(bitin[i]))

        ra = random.random()
        if check == 0: # if it is one output out1 with prob (1-pri_para)
            if ra > self.pri_para:
                return out1
            else:
                return out2
        else: # else output out2 with prob (1-pri_para)
            if ra > self.pri_para:
                return out2
            else:
                return out1       
     
    def encode_string(self,in_list):  # encode string into a privatized string
        out_list = [self.encode_symbol(x) for x in in_list] # encode each symbol in the string
        return out_list
    
    def decode_string(self, out_list, iffast = 1, normalization = 0): # get the privatized string and learn the original distribution
        #normalization options: 0: clip and normalize(default)
        #                       1: simplex projection
        #                       else: no nomalization
        
        #iffast: 0 use fast hadamard transform time O(n  + k\log k)
        #        1 no fast hadamard tansform  time O(n + k^2)
        
        l = len(out_list) 
        count,edges = np.histogram(out_list,range(self.outsz+1))
        dist = count/float(l)
        
        if iffast == 1: #if we use fast hadamard transform
            dist_mid = FWHT_A(self.outsz, dist) #do fast hadamard transform to the frequency vector
            dist_S = (dist_mid[1:self.insz+1] + 1)/float(2) #get the frequencies of C_i
        else: #if we don't use fast hadamard transform
            num = [0]*self.insz
            for x in range(0,self.outsz):
            #print x
                for i in range(1, self.insz+1): #if output x is in C_i(locations in row i which is 1), add it to the count of C_i
                    if self.parity_check(i,x) == 0:
                        num[i-1] = num[i-1] + count[x]
            dist_S = np.array([float(x)/float(l) for x in num]) #get the frequencies of C_i
            
        dist = (2*dist_S*(1+self.exp)-(1+self.exp))/float(self.exp-1) #calculate the corresponding estimate for p_i
        if normalization == 0: 
            dist = probability_normalize(dist) #clip and normalize
        if normalization == 1:
            dist = project_probability_simplex(dist) #simplex projection
        
        return dist


#The Hadamard randomized response for all regimes (Modified Version)

class Hadamard_Rand_general:
    def __init__(self, absz, pri_para, encode_acc = 0): # absz: alphabet size, pri_para: privacy parameter
        self.insz = absz #input alphabet size k
        #self.outsz = int(math.pow(2,math.ceil(math.log(absz+1,2)))) #output alphabet size: smallest exponent of 2 which is bigger than k
        #self.outbit = int(math.ceil(math.log(absz+1,2))) #output bit length
        self.pri_para = 1/(1+math.exp(pri_para)) #flipping probability to maintain local privacy
        self.exp = math.exp(pri_para) #privacy parameter
        #self.initbit = int(math.floor(math.log(self.exp,2))) # number of bits indicating the block number 
        self.initbit = int(math.floor(math.log(min(2*absz,self.exp),2))) # number of bits indicating the block number
        self.part = int(math.pow(2,self.initbit)) #total number of blocks B
        self.tailbit = int(math.ceil(math.log(float(self.insz)/float(self.part)+1,2))) #number of bits indicating the location within a block
        self.partsz = int(math.pow(2,self.tailbit)) # size of each block b
        self.num_one = int(self.partsz/float(2))
        self.outbit = self.tailbit + self.initbit #total number of output bits
        self.partnum = int(math.ceil(float(self.insz)/float(self.partsz - 1)))
        self.outsz = int(self.partsz*self.partnum) # output alphabet size K
        self.permute, self.reverse = Random_Permutation(absz) #Initialize random permutation
        
        self.ifencode_acc = encode_acc #whether to accelerate encoding process
        if encode_acc == 1:
            self.H = Hadarmard_init(self.partsz) # initialize Hadarmard matrix
        
    def entry_check(self,x,y): #check if the reduced hadamard entry is one (return 0 for 1)
        x_bit = bin(x)[2:].zfill(self.outbit)
        y_bit = bin(y)[2:].zfill(self.outbit)
        for i in range(0,self.initbit): # check if they are in the same block, if not, return -1
            if x_bit[i] != y_bit[i]:
                return True
        check = False
        for i in range(self.initbit, self.outbit): #check whether the entry is one within the block
            check = check^(int(x_bit[i]) & int(y_bit[i]))
        return check
                                  
            
    def encode_symbol_rough(self,in_symbol):  # encode a single symbol into a privatized version 
        # we use coupling argument to do this
        part_index = int(in_symbol)//(self.partsz-1)
        part_pos = int(in_symbol)%(self.partsz-1)+1
        in_column = (part_index << self.tailbit) + part_pos #map the input x to the xth column with weight b/2
        #in_column = part_index * self.partsz + part_pos
        out1 = np.random.randint(0,self.outsz) #get a random number out1 in the output alphabet as a potential output
        ra = random.random()
        if ra < (2*self.part)/(2*self.part-1+self.exp): #with certain prob, output the same symbol as from uniform distribution
            return out1
        else:
            out_pos = out1 & (self.partsz - 1)
            #out_pos = out1 % self.partsz
            out1 =  out_pos + (part_index << self.tailbit) # map out1 to the same block as in_column while maintain the location within the block
            #out1 = out_pos + part_index*self.partsz
            if self.ifencode_acc == 1:
                check = self.H[part_pos][out_pos]
            else:
                check = 1 - self.entry_check(in_column,out1)

            if check == 0: #if location (in_column, out1) is one, output out1
                return out1
            else: #else flip bit at the left most location where bit representation of in_column is one 
                #bitin = bin(int(in_column))[2:].zfill(self.outbit)
                check = 1
                for i in range(self.outbit - self.initbit): 
                    if in_column%2 == 1:
                        #out1 = out1 ^ (pow(2,self.outbit - i -1))
                        out1 = out1 ^ check
                        break
                    in_column = in_column >> 1
                    check = check << 1
                return out1
            
    #delete the first row of each block
    def encode_symbol(self, in_symbol):
        while(1):
            out = self.encode_symbol_rough(in_symbol)
            if out%self.partsz != 0:
                return out
    
    def encode_string(self,in_list):  # encode string into a privatized string
        out_list = [self.encode_symbol(self.permute[x]) for x in in_list]
        return out_list
    
    
    def decode_string(self, out_list, iffast = 1, normalization = 0): # get the privatized string and learn the original distribution
        #normalization options: 0: clip and normalize(default)
        #                       1: simplex projection
        #                       else: no nomalization
        
        #iffast: 0 use fast hadamard transform time O(n  + k\log k)
        #        1 no fast hadamard tansform  time O(n + k^2)
        
        l = len(out_list)
        count,edges = np.histogram(out_list,range(self.outsz+1))
        freq = count/float(l)
        
        
        if iffast == 1:
            #parts = self.insz//(self.partsz-1) 
            freq_S = np.zeros(self.outsz)
            freq_block = np.zeros(self.partnum)
            for i in range(0, self.partnum):
                Trans = FWHT_A(self.partsz, freq[i*self.partsz: (i+1)*self.partsz])
                freq_block[i] = Trans[0]
                freq_S[i*(self.partsz-1): (i+1)*(self.partsz-1)] = ( - Trans[1:self.partsz] + Trans[0])/float(2)         
            dist_S = freq_S[0:self.insz]
            
        else:
            freq_block = np.zeros(self.part) # count the number of appearances of each block
            for i in range(0,self.part): 
                #count_block[i] = np.sum(count[i*self.partsz : (i+1)*self.partsz - 1])
                for j in range(0,self.partsz):
                    freq_block[i] = freq_block[i] + freq[i*self.partsz+j]
            #freq_block = np.true_divide(count_block,l) # calculate the frequency of each block
            #dist_block = np.true_divide((2*self.part-1+self.exp)*(freq_block)-2,self.exp-1) # calculate the estimated original prob of each block                    
            for i in range(0, self.insz): 
                pi = int(i)//(self.partsz-1)
                ti = pi*self.partsz + int(i)%(self.partsz-1)+1
                for x in range(pi*self.partsz, (pi+1)*self.partsz): # count the number of appearances of each C_i
                    if self.entry_check(ti,x) == 0:
                        dist_S[i] = dist_S[i] + freq[x]
                        
        lbd = float(self.outsz - self.partnum)/float(self.num_one)
        c1 = lbd-1+self.exp
        
        dist_block = np.true_divide(c1*(freq_block)- 2 + 1/float(self.num_one),self.exp-1) # calculate the estimated original prob of each block
        
        c2 = self.exp - 1
        #dist = [float(2*c1*dist_S[i] - c2*dist_block[i//(self.partsz-1)] - 2)/float(c3) for i in range(0,self.insz) ]
        dist = [float(2*c1*dist_S[i] - c2*dist_block[i//(self.partsz-1)] - 2)/float(c2) for i in range(0,self.insz) ]
        
        if normalization == 0: 
            dist = probability_normalize(dist) #clip and normalize
        if normalization == 1:
            dist = project_probability_simplex(dist) #simplex projection
        
        #reverse the permuation
        dist1 = np.zeros(self.insz)
        for i in range(self.insz):
            dist1[int(self.reverse[i])] = dist[i]
        return dist1
    
    
    def decode_string_old(self, out_list): # get the privatized string and learn the original distribution
        
        l = len(out_list)
        dist_S = np.zeros(self.insz)
        count,edges = np.histogram(out_list,range(self.outsz+1))
        freq = count/float(l)
        
        freq_block = np.zeros(self.part) # count the number of appearances of each block
        for i in range(0,self.part): 
            #count_block[i] = np.sum(count[i*self.partsz : (i+1)*self.partsz - 1])
            for j in range(0,self.partsz):
                freq_block[i] = freq_block[i] + freq[i*self.partsz+j]
        
        
        #freq_block = np.true_divide(count_block,l) # calculate the frequency of each block
        dist_block = np.true_divide((2*self.part-1+self.exp)*(freq_block)-2,self.exp-1) # calculate the estimated original prob of each block
                    
        for i in range(0, self.insz): 
            pi = int(i)//(self.partsz-1)
            ti = pi*self.partsz + int(i)%(self.partsz-1)+1
            for x in range(pi*self.partsz, (pi+1)*self.partsz): # count the number of appearances of each C_i
                if self.entry_check(ti,x) == 0:
                    dist_S[i] = dist_S[i] + freq[x]

        #dist_S = np.zeros(self.insz)
        #dist_S = np.true_divide(num,l) #calculate the frequency of each C_i
        dist_inter = np.true_divide(2*(dist_S*(2*self.part-1+self.exp)-1),self.exp-1) # calculate intermediate prob
        dist = [dist_inter[i] - dist_block[i//(self.partsz-1)] for i in range(0,self.insz)] # calculate the estimated prob for each symbol
        dist = np.maximum(dist,0) #map it to be positive
        norm = np.sum(dist)
        dist = np.true_divide(dist,norm) #ensure the l_1 norm is one
        return dist
    
    #def decode_string_normalize(self, out_list): #normalized outputs using clip and normalize
    #    dist = self.decode_string_permute(out_list)
    #    dist = probability_normalize(dist)
    #    return dist
    
    #def decode_string_project(self, out_list): #projected outputs
    #    dist = self.decode_string_permute(out_list)
    #    dist = project_probability_simplex(dist)
    #    return dist
    
    #def decode_string_permute(self, out_list): # get the privatized string and learn the original distribution
    #    dist1 = self.decode_string_fast(out_list)
    #    dist = np.zeros(self.insz)
    #    for i in range(self.insz):
    #        dist[int(self.reverse[i])] = dist1[i]
    #    return dist

#The Hadamard randomized response for all regimes (original version)
class Hadamard_Rand_general_original:
    def __init__(self, absz, pri_para, encode_acc = 0): # absz: alphabet size, pri_para: privacy parameter
        #set encode_acc = 1 to enable fast encoding by intializing hadamard matrix
        
        self.insz = absz #input alphabet size k
        #self.outsz = int(math.pow(2,math.ceil(math.log(absz+1,2)))) #output alphabet size: smallest exponent of 2 which is bigger than k
        #self.outbit = int(math.ceil(math.log(absz+1,2))) #output bit length
        self.pri_para = 1/(1+math.exp(pri_para)) #flipping probability to maintain local privacy
        self.exp = math.exp(pri_para) #privacy parameter
        #self.initbit = int(math.floor(math.log(self.exp,2))) # number of bits indicating the block number 
        self.initbit = int(math.floor(math.log(min(2*absz,self.exp),2))) # number of bits indicating the block number
        self.part = int(math.pow(2,self.initbit)) #number of blocks B
        self.tailbit = int(math.ceil(math.log(float(self.insz)/float(self.part)+1,2))) #number of bits indicating the location within a block
        self.partsz = int(math.pow(2,self.tailbit)) # size of each block b
        self.outbit = self.tailbit + self.initbit #total number of output bits
        self.outsz = int(math.pow(2,self.outbit)) # output alphabet size K
        self.permute, self.reverse = Random_Permutation(absz) #Initialize random permutation
        # self.permute,self.reverse = [], []
        self.dist_inter = []
        self.dist_block = []
        self.ifencode_acc = encode_acc #whether to accelerate encoding process
        if encode_acc == 1:
            self.H = Hadarmard_init(self.partsz) # initialize Hadarmard matrix
        
        
    def entry_check(self,x,y): #check if the reduced hadamard entry is one (return 0 for 1)
        x_bit = bin(x)[2:].zfill(self.outbit)
        y_bit = bin(y)[2:].zfill(self.outbit)
        for i in range(0,self.initbit): # check if they are in the same block, if not, return -1
            if x_bit[i] != y_bit[i]:
                return True
        check = False
        for i in range(self.initbit, self.outbit): #check whether the entry is one within the block
            check = check^(int(x_bit[i]) & int(y_bit[i]))
        return check
                                  
    
    def encode_symbol(self,in_symbol):  # encode a single symbol into a privatized version 
        # we use coupling argument to do this
        part_index = int(in_symbol)//(self.partsz-1)
        part_pos = int(in_symbol)%(self.partsz-1)+1
        in_column = (part_index << self.tailbit) + part_pos #map the input x to the xth column with weight b/2
        #in_column = part_index * self.partsz + part_pos
        out1 = np.random.randint(0,self.outsz) #get a random number out1 in the output alphabet as a potential output
        ra = random.random()
        if ra < (2*self.part)/(2*self.part-1+self.exp): #with certain prob, output the same symbol as from uniform distribution
            return out1
        else:
            out_pos = out1 & (self.partsz - 1)
            #out_pos = out1 % self.partsz
            out1 =  out_pos + (part_index << self.tailbit) # map out1 to the same block as in_column while maintain the location within the block
            #out1 = out_pos + part_index*self.partsz
            if self.ifencode_acc == 1:
                check = 1 - self.H[part_pos][out_pos]
            else:
                check = self.entry_check(in_column,out1)

            if check == 0: #if location (in_column, out1) is one, output out1
                return out1
            else: #else flip bit at the left most location where bit representation of in_column is one 
                #bitin = bin(int(in_column))[2:].zfill(self.outbit)
                check = 1
                for i in range(self.outbit - self.initbit): 
                    if in_column%2 == 1:
                        #out1 = out1 ^ (pow(2,self.outbit - i -1))
                        out1 = out1 ^ check
                        break
                    in_column = in_column >> 1
                    check = check << 1
                return out1
    
    def encode_string(self,in_list): #permute before encoding
        out_list = [self.encode_symbol(self.permute[x]) for x in in_list]
        return out_list        
    
    
    def decode_string(self, out_list,iffast = 1, normalization = 0): # get the privatized string and learn the original distribution
        #normalization options: 0: clip and normalize(default)
        #                       1: simplex projection
        #                       else: no nomalization
        
        #iffast: 0 use fast hadamard transform time O(n  + k\log k)
        #        1 no fast hadamard tansform  time O(n + kB), B is the block size
        
        l = len(out_list)
        count,edges = np.histogram(out_list,range(self.outsz+1))
        freq = count/float(l)
        
        if iffast == 1:
            parts = self.insz//(self.partsz-1)
            freq_S = np.zeros((parts+1)*self.partsz)
            freq_block = np.zeros((parts+1)*self.partsz)
        
            for i in range(0, parts+1):

                Trans = FWHT_A(self.partsz, freq[i*self.partsz: (i+1)*self.partsz])
                freq_block[i] = Trans[0]
                freq_S[i*(self.partsz-1): (i+1)*(self.partsz-1)] = (Trans[1:self.partsz] + Trans[0])/float(2) 
            dist_S = freq_S[0:self.insz]
        
            dist_block = np.true_divide((2*self.part-1+self.exp)*(freq_block)-2,self.exp-1) # calculate the estimated original prob of each block
        
        else:
            freq_block = np.zeros(self.part) # count the number of appearances of each block
            for i in range(0,self.part): 
                #count_block[i] = np.sum(count[i*self.partsz : (i+1)*self.partsz - 1])
                for j in range(0,self.partsz):
                    freq_block[i] = freq_block[i] + freq[i*self.partsz+j]     
                    
            dist_block = np.true_divide((2*self.part-1+self.exp)*(freq_block)-2,self.exp-1) # calculate the estimated original prob of each block
            dist_S = np.zeros(self.insz)
            for i in range(0, self.insz): 
                pi = int(i)//(self.partsz-1)
                ti = pi*self.partsz + int(i)%(self.partsz-1)+1
                for x in range(pi*self.partsz, (pi+1)*self.partsz): # count the number of appearances of each C_i
                    if self.entry_check(ti,x) == 0:
                        dist_S[i] = dist_S[i] + freq[x]
        
        dist_inter = np.true_divide(2*(dist_S*(2*self.part-1+self.exp)-1),self.exp-1) # calculate intermediate prob
        dist = [dist_inter[i] - dist_block[i//(self.partsz-1)] for i in range(0,self.insz)] # calculate the estimated prob for each symbol
        
        if normalization == 0: 
            dist = probability_normalize(dist) #clip and normalize
        if normalization == 1:
            dist = project_probability_simplex(dist) #simplex projection
        
        #reverse the permuation
        dist1 = np.zeros(self.insz)
        for i in range(self.insz):
            dist1[int(self.reverse[i])] = dist[i]
        return dist1

    # Debugging methods, generate_dist is the first half of decode_string
    def generate_dist(self, out_list, iffast = 1, normalization = 0):
        l = len(out_list)
        count,edges = np.histogram(out_list,range(self.outsz+1))
        freq = count/float(l)

        if iffast == 1:
            parts = self.insz//(self.partsz-1)
            freq_S = np.zeros((parts+1)*self.partsz)
            freq_block = np.zeros((parts+1)*self.partsz)

            for i in range(0, parts+1):
                Trans = FWHT_A(self.partsz, freq[i*self.partsz: (i+1)*self.partsz])
                freq_block[i] = Trans[0]
                freq_S[i*(self.partsz-1): (i+1)*(self.partsz-1)] = (Trans[1:self.partsz] + Trans[0])/float(2)
            dist_S = freq_S[0:self.insz]

            dist_block = np.true_divide((2*self.part-1+self.exp)*(freq_block)-2,self.exp-1) # calculate the estimated original prob of each block

        else:
            freq_block = np.zeros(self.part) # count the number of appearances of each block
            for i in range(0,self.part):
                #count_block[i] = np.sum(count[i*self.partsz : (i+1)*self.partsz - 1])
                for j in range(0,self.partsz):
                    freq_block[i] = freq_block[i] + freq[i*self.partsz+j]

            dist_block = np.true_divide((2*self.part-1+self.exp)*(freq_block)-2,self.exp-1) # calculate the estimated original prob of each block
            dist_S = np.zeros(self.insz)
            for i in range(0, self.insz):
                pi = int(i)//(self.partsz-1)
                ti = pi*self.partsz + int(i)%(self.partsz-1)+1
                for x in range(pi*self.partsz, (pi+1)*self.partsz): # count the number of appearances of each C_i
                    if self.entry_check(ti,x) == 0:
                        dist_S[i] = dist_S[i] + freq[x]

        dist_inter = np.true_divide(2*(dist_S*(2*self.part-1+self.exp)-1),self.exp-1) # calculate intermediate prob
        self.dist_block = dist_block
        self.dist_inter = dist_inter


    def estimate(self, item):
        return self.dist_inter[item] - self.dist_block[item//(self.partsz-1)]

