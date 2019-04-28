IMAGE_DIR = '../../datasets/super-resolution/BSD100';
SR_RES = '../SRCNN/result.mat';
upscale_factor = 2;


image_filenames = dir(IMAGE_DIR);

im_sr_res = load(SR_RES);

avg_psnr_bic = 0;
avg_psnr_sr = 0;
cnt = 0

for i = 1:length(image_filenames)
    filename = image_filenames(i).name;
    [path, f, ext] = fileparts(filename);
    if strcmp(filename, '.') == 1 || strcmp(filename, '..') == 1 ...
            || (strcmp(ext, '.jpg') == 0 && strcmp(ext, '.jpeg') == 0 && strcmp(ext, '.png') == 0)
        continue
    end   
    cnt = cnt + 1;
    fprintf('%s\n', f);
    image_filename = fullfile(IMAGE_DIR, filename);
    im = imread(image_filename);
    fprintf('image size: (%d, %d)\n', size(im, 1), size(im, 2));

    if size(im, 3) > 1
        im = rgb2ycbcr(im);
        im = im(:, :, 1);
    end

    im = modcrop(im, upscale_factor);
    im = single(im) / 255.;

    % bicubic interpolation
    im_lr = imresize(im, 1.0/upscale_factor, 'bicubic');
    im_bicubic = imresize(im_lr, upscale_factor, 'bicubic');
    
    % srcnn
    im_sr = getfield(im_sr_res, strcat('a', f));
    
    % remove border
    im_sr = shave(uint8(im_sr * 255), [upscale_factor, upscale_factor]);
    im = shave(uint8(im * 255), [upscale_factor, upscale_factor]);
    im_bicubic = shave(uint8(im_bicubic * 255), [upscale_factor, upscale_factor]);
    
    % compute PSNR
    psnr_bic = compute_psnr(im, im_bicubic);
    psnr_sr = compute_psnr(im, im_sr);
    avg_psnr_bic = avg_psnr_bic + psnr_bic;
    avg_psnr_sr = avg_psnr_sr + psnr_sr;
    
    fprintf('bicubic psnr: %.4f\n', psnr_bic);
    fprintf('SRCNN psnr: %.4f\n', psnr_sr);
    fprintf('\n');
end

avg_psnr_bic = avg_psnr_bic / cnt;
avg_psnr_sr = avg_psnr_sr / cnt;

fprintf('Average bicubic PSNR: %.4f\n', avg_psnr_bic);
fprintf('Average SR PSNR: %.4f\n', avg_psnr_sr);


