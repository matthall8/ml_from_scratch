#%%
import numpy as np # linear algebra

class Conv_Op:
    
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size*filter_size)
        
    def image_region(self, image):
        height, width = image.shape
        self.image = image
        for j in range(height - self.filter_size+1):
            for k in range(width-self.filter_size+1):
                image_patch = image[j:(j+self.filter_size), k:(k+self.filter_size)]
                yield image_patch,j,k
    
    def forward_prop(self, image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size+1, width - self.filter_size+1, self.num_filters))
        for image_patch, i, j in self.image_region(image):
            conv_out[i,j] = np.sum(image_patch *self.conv_filter, axis=(1,2))
        return conv_out
    
    
    def back_prop(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)
        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k] += image_patch * dL_dout[i,j,k]
    
        self.conv_filter -= learning_rate*dL_dF_params
        return dL_dF_params

class Max_Pool:
    
    def __init__(self, filter_size):
        self.filter_size = filter_size
        
    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image
        
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[(i*self.filter_size) : (i*self.filter_size+self.filter_size),  (j*self.filter_size) : (j*self.filter_size+self.filter_size)]
                yield image_patch,i,j
    
    def forward_prop(self, image):
        height, width, num_filters = image.shape
        output = np.zeros((height // self.filter_size, width // self.filter_size, num_filters))
        
        for image_patch, i, j in self.image_region(image):
            output[i,j] = np.amax(image_patch, axis=(0,1))
            
        return output
    
    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch, axis=(0,1))
            
            for il in range(height):
                for jl in range(width):
                    for kl in range(num_filters):
                        if image_patch[il, jl, kl] == maximum_val[kl]:
                            dL_dmax_pool[i*self.filter_size + il, j*self.filter_size + jl,  kl]
                            
    
            return dL_dmax_pool

class Softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node,softmax_node)/input_node
        self.bias = np.random.randn(softmax_node)
            
    def forward_prop(self, image):
            
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return exp_out / np.sum(exp_out, axis = 0)
        
    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue
                
            transformation_eq = np.exp(self.out)
            s_total = np.sum(transformation_eq)
                
            # Gradients. with respect to out (z)
                
            dy_dz = -transformation_eq[i] * transformation_eq / (s_total ** 2)
            dy_dz[i] = transformation_eq[i] * (s_total - transformation_eq[i]) / (s_total ** 2)
                
            # Gradients of totals against weights / biases / input
                
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_input = self.weight
                
            # Gradients of loss against totals
            dL_dz = grad * dy_dz
                
            # Gradients of loss against weights / biases / inputs
                
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_input @ dL_dz
                
        # update weights and biases
                
        self.weight -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
                
        return dL_d_inp.reshape(self.orig_im_shape)