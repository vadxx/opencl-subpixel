__local int modul(int a,  int b)
{
    return ((a % b) + b) % b;
}

__kernel void sub_pixel(__global float *matr,
                        __global float *buff,
                        const int direction,
                        const int pixelX,
                        const int pixelY,
                        const int rows,
                        const int cols
                        )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float delta = 0.0f;
    
    //  Direction: 0 - vertical; 1 - horizontal;
    int index_pix0, index_pix_left, index_pix_right;     // indexes in response
    float pix0, pix_left, pix_right;                    // values in response

    if (direction == 0)
    {
        // neighbouring rows
        index_pix_left = modul(pixelY - 1, rows);
        index_pix_right = modul(pixelY + 1, rows);
        
        pix0 = matr[index_pix0 * cols + pixelX];
        pix_left = matr[index_pix_left * cols + pixelX];
        pix_right = matr[index_pix_right * cols + pixelX];
    }
    else if (direction == 1)
    {
        // neighbouring cols
        index_pix_left = modul(pixelX - 1, cols);
        index_pix_right = modul(pixelX + 1, cols);
        
        pix0 = matr[pixelY * cols + index_pix0];
        pix_left = matr[pixelY * cols + index_pix_left];
        pix_right = matr[pixelY * cols + index_pix_right];
    }

    delta = 0.5f * (pix_right - pix_left) / (2 * pix0 - pix_right - pix_left);
    if (!isfinite(delta))
    {
        delta = 0;
    }
    //Result checking
    buff[0] = delta;
    buff[1] = (float)pixelX;
    buff[2] = (float)pixelY;
    
}
