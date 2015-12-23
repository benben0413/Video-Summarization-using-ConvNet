import skimage.transform
import theano.tensor as s

def upsample4D(input_image,upscale,batch_size,image_size):
    d4=[]
    for i in xrange(batch_size):
        d3_image=input_image[i]
        d3=[]
        for j in xrange(256):
            d2_image=skimage.transform.pyramid_expand(d3_image[j],upscale=upscale,sigma=None, order=1, mode='reflect', cval=0)
            d3.append(d2_image.reshape((1,image_size,image_size)))
        d3_out=s.concatenate(d3,axis=0)
        d4.append(d3_out.reshape((1,256,image_size,image_size)))
    d4_out=s.concatenate(d4,axis=0)
    return d4_out
                  
        
    
