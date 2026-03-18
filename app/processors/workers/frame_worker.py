# frame_worker.py
import traceback
from typing import TYPE_CHECKING
import threading
from math import floor, ceil
import torch
from skimage import transform as trans
from torchvision.transforms import v2
import torchvision
from torchvision import transforms
import numpy as np
from app.processors.utils import faceutil
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.helpers.miscellaneous import t512,t384,t256,t128, ParametersDict
if TYPE_CHECKING:
from app.ui.main_ui import MainWindow
torchvision.disable_beta_transforms_warning()

class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window: 'MainWindow', frame_number, frame_queue, is_single_frame=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame = frame
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.parameters = {}
        self.target_faces = main_window.target_faces
        self.compare_images = []
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False

    def run(self):
        try:
            # Update parameters from markers (if exists) without concurrent access from other threads
            with self.main_window.models_processor.model_lock: 
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
            self.parameters = self.main_window.parameters.copy()
            # Check if view mask or face compare checkboxes are checked
            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked() 
            self.is_view_face_mask  = self.main_window.faceMaskCheckBox.isChecked() 

            # Process the frame with model inference
            if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked() or self.main_window.control['FrameEnhancerEnableToggle']:
                self.frame = self.process_frame()
            else:
                # Img must be in BGR format
                self.frame = self.frame[..., ::-1]  # Swap the channels from RGB to BGR
            self.frame = np.ascontiguousarray(self.frame)

            # Display the frame if processing is still active
            pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

            # Output processed Webcam frame
            if self.video_processor.file_type=='webcam' and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)

            #Output Video frame (while playing)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
            # Output Image/Video frame (Single frame)
            else:
                self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)

            # Mark the frame as done in the queue
            self.video_processor.frame_queue.get()
            self.video_processor.frame_queue.task_done()

            # Check if playback is complete
            if self.video_processor.frame_queue.empty() and not self.video_processor.processing and self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e:
            print(f"Error in FrameWorker: {e}")
            traceback.print_exc()

    def process_frame(self):
        # Load frame into VRAM
        img = torch.from_numpy(self.frame.astype('uint8')).to(self.models_processor.device)
        img = img.permute(2,0,1)

        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]

        if img_x <512 and img_y <512:
            if img_x <= img_y:
                new_height = int(512*img_y/img_x)
                tscale = v2.Resize((new_height, 512), antialias=True)
            else:
                new_height = 512
                tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)
        elif img_x <512:
            new_height = int(512*img_y/img_x)
            tscale = v2.Resize((new_height, 512), antialias=True)
            img = tscale(img)
        elif img_y <512:
            new_height = 512
            tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)

        control = self.main_window.control.copy()
        
        # Rotate the frame
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        use_landmark_detection=control['LandmarkDetectToggle']
        landmark_detect_mode=control['LandmarkDetectModelSelection']
        from_points = control["DetectFromPointsToggle"]
        
        if self.main_window.editFacesButton.isChecked():
            if not use_landmark_detection or landmark_detect_mode=="5":
                use_landmark_detection = True
                landmark_detect_mode = "203"
            from_points = True

        bboxes, kpss_5, kpss = self.models_processor.run_detect(
            img, control['DetectorModelSelection'], 
            max_num=control['MaxFacesToDetectSlider'], 
            score=control['DetectorScoreSlider']/100.0, 
            input_size=(512, 512), 
            use_landmark_detection=use_landmark_detection, 
            landmark_detect_mode=landmark_detect_mode, 
            landmark_score=control["LandmarkDetectScoreSlider"]/100.0, 
            from_points=from_points, 
            rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
        )
        
        det_faces_data = []
        if len(kpss_5) >0:
            for i in range(kpss_5.shape[0]):
                face_kps_5 = kpss_5[i]
                face_kps_all = kpss[i]
                face_emb, _ = self.models_processor.run_recognize_direct(
                    img, face_kps_5, 
                    control['SimilarityTypeSelection'], 
                    control['RecognitionModelSelection']
                )
                det_faces_data.append({'kps_5': face_kps_5, 'kps_all': face_kps_all, 'embedding': face_emb, 'bbox': bboxes[i]})

        compare_mode = self.is_view_face_mask or self.is_view_face_compare
        
        if det_faces_data:
            for i, fface in enumerate(det_faces_data):
                for _, target_face in self.main_window.target_faces.items():
                    parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters)

                    if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked():
                        sim = self.models_processor.findCosineDistance(
                            fface['embedding'], 
                            target_face.get_embedding(control['RecognitionModelSelection'])
                        )
                        if sim >= parameters['SimilarityThresholdSlider']:
                            s_e = None
                            fface['kps_5'] = self.keypoints_adjustments(fface['kps_5'], parameters)
                            arcface_model = self.models_processor.get_arcface_model(parameters['SwapModelSelection'])
                            dfm_model = parameters['DFMModelSelection']
                            
                            if self.main_window.swapfacesButton.isChecked():
                                if parameters['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                    s_e = target_face.assigned_input_embedding.get(arcface_model, None)
                                if s_e is not None and np.isnan(s_e).any():
                                    s_e = None
                            else:
                                dfm_model = None
                                s_e = None

                            img, fface['original_face'], fface['swap_mask'] = self.swap_core(
                                img, fface['kps_5'], s_e=s_e, t_e=target_face.get_embedding(arcface_model), 
                                parameters=parameters, control=control, dfm_model=dfm_model
                            )
                                    
                            if self.main_window.editFacesButton.isChecked():
                                img = self.swap_edit_face_core(img, fface['kps_all'], parameters, control)

        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=-control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        if control['ShowAllDetectedFacesBBoxToggle']:
            img = self.draw_bounding_boxes_on_detected_faces(img, det_faces_data, control)

        if control["ShowLandmarksEnableToggle"] and det_faces_data:
            img = img.permute(1,2,0)
            img = self.paint_face_landmarks(img, det_faces_data, control)
            img = img.permute(2,0,1)

        if compare_mode: 
            img = self.get_compare_faces_image(img, det_faces_data, control)

        if control['FrameEnhancerEnableToggle'] and not compare_mode:
            img = self.enhance_core(img, control=control)

        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        return img[..., ::-1]

    def keypoints_adjustments(self, kps_5: np.ndarray, parameters: dict) -> np.ndarray:
        if parameters['FaceAdjEnableToggle']:
            kps_5[:,0] += parameters['KpsXSlider']
            kps_5[:,1] += parameters['KpsYSlider']
            kps_5[:,0] -= 255
            kps_5[:,0] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,0] += 255
            kps_5[:,1] -= 255
            kps_5[:,1] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,1] += 255

        if parameters['LandmarksPositionAdjEnableToggle']:
            kps_5[0][0] += parameters['EyeLeftXAmountSlider']
            kps_5[0][1] += parameters['EyeLeftYAmountSlider']
            kps_5[1][0] += parameters['EyeRightXAmountSlider']
            kps_5[1][1] += parameters['EyeRightYAmountSlider']
            kps_5[2][0] += parameters['NoseXAmountSlider']
            kps_5[2][1] += parameters['NoseYAmountSlider']
            kps_5[3][0] += parameters['MouthLeftXAmountSlider']
            kps_5[3][1] += parameters['MouthLeftYAmountSlider']
            kps_5[4][0] += parameters['MouthRightXAmountSlider']
            kps_5[4][1] += parameters['MouthRightYAmountSlider']
        return kps_5

    def paint_face_landmarks(self, img: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        p = 2
        for i, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                parameters = self.parameters[target_face.face_id]
                sim = self.models_processor.findCosineDistance(
                    fface['embedding'], 
                    target_face.get_embedding(control['RecognitionModelSelection'])
                )
                if sim >= parameters['SimilarityThresholdSlider']:
                    if parameters['LandmarksPositionAdjEnableToggle']:
                        kcolor = tuple((255, 0, 0))
                        keypoints = fface['kps_5']
                    else:
                        kcolor = tuple((0, 255, 255))
                        keypoints = fface['kps_all']

                    for kpoint in keypoints:
                        for i in range(-1, p):
                            for j in range(-1, p):
                                try:
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][0] = kcolor[0]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][1] = kcolor[1]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][2] = kcolor[2]
                                except ValueError:
                                    continue
        return img

    def draw_bounding_boxes_on_detected_faces(self, img: torch.Tensor, det_faces_data: list, control: dict):
        for i, fface in enumerate(det_faces_data):
            color = [0, 255, 0]
            bbox = fface['bbox']
            x_min, y_min, x_max, y_max = map(int, bbox)
            _, h, w = img.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
            max_dimension = max(img.shape[1], img.shape[2])
            thickness = max(4, max_dimension // 400)
            color_tensor = torch.tensor(color, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            img[:, y_min:y_min + thickness, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            img[:, y_max - thickness + 1:y_max + 1, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            img[:, y_min:y_max + 1, x_min:x_min + thickness] = color_tensor.expand(-1, y_max - y_min + 1, thickness)
            img[:, y_min:y_max + 1, x_max - thickness + 1:x_max + 1] = color_tensor.expand(-1, y_max - y_min + 1, thickness)   
        return img

    def get_compare_faces_image(self, img: torch.Tensor, det_faces_data: dict, control: dict) -> torch.Tensor:
        imgs_to_vstack = []
        for _, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                parameters = self.parameters[target_face.face_id]
                sim = self.models_processor.findCosineDistance(
                    fface['embedding'], 
                    target_face.get_embedding(control['RecognitionModelSelection'])
                )
                if sim >= parameters['SimilarityThresholdSlider']:
                    modified_face = self.get_cropped_face_using_kps(img, fface['kps_5'], parameters)
                    if control['FrameEnhancerEnableToggle']:
                        modified_face_enhance = self.enhance_core(modified_face, control=control)
                        modified_face_enhance = modified_face_enhance.float() / 255.0
                        modified_face = torch.functional.F.interpolate(
                            modified_face_enhance.unsqueeze(0),
                            size=modified_face.shape[1:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        modified_face = (modified_face * 255).clamp(0, 255).to(dtype=torch.uint8)
                    imgs_to_cat = []
                    if fface['original_face'] is not None:
                        imgs_to_cat.append(fface['original_face'].permute(2, 0, 1))
                    imgs_to_cat.append(modified_face)
                    if fface['swap_mask'] is not None:
                        fface['swap_mask'] = 255-fface['swap_mask']
                        imgs_to_cat.append(fface['swap_mask'].permute(2, 0, 1))
                    img_compare = torch.cat(imgs_to_cat, dim=2)
                    imgs_to_vstack.append(img_compare)

        if imgs_to_vstack:
            max_width = max(img_to_stack.size(2) for img_to_stack in imgs_to_vstack)
            padded_imgs = [
                torch.nn.functional.pad(img_to_stack, (0, max_width - img_to_stack.size(2), 0, 0)) 
                for img_to_stack in imgs_to_vstack
            ]
            img_vstack = torch.cat(padded_imgs, dim=1)
            img = img_vstack
        return img
    
    def get_cropped_face_using_kps(self, img: torch.Tensor, kps_5: np.ndarray, parameters: dict) -> torch.Tensor:
        tform = self.get_face_similarity_tform(parameters['SwapModelSelection'], kps_5)
        face_512 = v2.functional.affine(
            img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]), 
            tform.scale, 0, center=(0,0), interpolation=v2.InterpolationMode.BILINEAR
        )
        face_512 = v2.functional.crop(face_512, 0,0, 512, 512)
        return face_512

    def get_face_similarity_tform(self, swapper_model: str, kps_5: np.ndarray) -> trans.SimilarityTransform:
        tform = trans.SimilarityTransform()
        if swapper_model not in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS', 'HyperSwap_1C_256'):
            dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst = np.squeeze(dst)
            tform.estimate(kps_5, dst)
        elif swapper_model == "CSCS":
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M
        return tform
  
    def get_transformed_and_scaled_faces(self, tform, img):
        original_face_512 = v2.functional.affine(
            img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]), 
            tform.scale, 0, center=(0,0), interpolation=v2.InterpolationMode.BILINEAR
        )
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        return original_face_512, original_face_384, original_face_256, original_face_128

    def get_affined_face_dim_and_swapping_latents(self, original_faces, swapper_model, dfm_model, s_e, t_e, parameters):
        original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        
        print(f"[DEBUG] get_affined_face_dim_and_swapping_latents: Модель={swapper_model}")
        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
            latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim = 1
            if parameters['SwapperResSelection'] == '128':
                dim = 1
                input_face_affined = original_face_128
            elif parameters['SwapperResSelection'] == '256':
                dim = 2
                input_face_affined = original_face_256
            elif parameters['SwapperResSelection'] == '384':
                dim = 3
                input_face_affined = original_face_384
            elif parameters['SwapperResSelection'] == '512':
                dim = 4
                input_face_affined = original_face_512

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            self.models_processor.load_inswapper_iss_emap(swapper_model)
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(s_e, version)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(t_e, version)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'HyperSwap_1C_256':
            print("[DEBUG] get_affined_face_dim_and_swapping_latents: HyperSwap_1C_256 - загрузка emap...")
            self.models_processor.load_hyperswap_emap()
            print("[DEBUG] get_affined_face_dim_and_swapping_latents: HyperSwap_1C_256 - вычисление латента...")
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_hyperswap(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_hyperswap(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim = 2
            input_face_affined = original_face_256
            print(f"[DEBUG] get_affined_face_dim_and_swapping_latents: HyperSwap_1C_256 - dim={dim}, latent shape={latent.shape}")

        elif swapper_model == 'SimSwap512':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim = 4
            input_face_affined = original_face_512

        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'CSCS':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            dfm_model = self.models_processor.load_dfm_model(dfm_model)
            latent = []
            input_face_affined = original_face_512
            dim = 4
        else:
            latent = None
            input_face_affined = original_face_256
            dim = 2
            
        return input_face_affined, dfm_model, dim, latent

    def get_swapped_and_prev_face(self, output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters):
        prev_face = input_face_affined.clone()
        
        if swapper_model == 'Inswapper128':
            with torch.no_grad():
                for _ in range(itex):
                    for j in range(dim):
                        for i in range(dim):
                            input_face_disc = input_face_affined[j::dim, i::dim]
                            input_face_disc = input_face_disc.permute(2, 0, 1)
                            input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                            swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device=self.models_processor.device).contiguous()
                            self.models_processor.run_inswapper(input_face_disc, latent, swapper_output, sharpness=parameters['SwapperSharpnessDecimalSlider'])
                            swapper_output = torch.squeeze(swapper_output)
                            swapper_output = swapper_output.permute(1, 2, 0)
                            output[j::dim, i::dim] = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            with torch.no_grad():
                for _ in range(itex):
                    input_face_disc = input_face_affined.permute(2, 0, 1)
                    input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                    swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                    self.models_processor.run_iss_swapper(input_face_disc, latent, swapper_output, version)
                    swapper_output = torch.squeeze(swapper_output)
                    swapper_output = swapper_output.permute(1, 2, 0)
                    output = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == 'HyperSwap_1C_256':
            with torch.no_grad():
                for _ in range(itex):
                    input_face_disc = input_face_affined.permute(2, 0, 1)
                    input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                    swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                    # Получаем маску
                    mask_tensor = self.models_processor.run_hyperswap(input_face_disc, latent, swapper_output)
                    swapper_output = torch.squeeze(swapper_output)
                    swapper_output = swapper_output.permute(1, 2, 0)
                    output = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == 'SimSwap512':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_simswap512(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
            for k in range(itex):
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = torch.sub(input_face_disc, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_ghostface(input_face_disc, latent, swapper_output, swapper_model)
                swapper_output = swapper_output[0]
                swapper_output = swapper_output.permute(1, 2, 0)
                swapper_output = torch.mul(swapper_output, 127.5)
                swapper_output = torch.add(swapper_output, 127.5)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                input_face_affined = torch.div(input_face_affined, 255)
                output = swapper_output.clone()
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'CSCS':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_cscs(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)
    
        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            out_celeb, _, _ = dfm_model.convert(original_face_512, parameters['DFMAmpMorphSlider']/100, rct=parameters['DFMRCTColorToggle'])
            prev_face = input_face_affined.clone()
            input_face_affined = out_celeb.clone()
            output = out_celeb.clone()

        output = output.permute(2, 0, 1)
        swap = t512(output)   
        return swap, prev_face

    def get_border_mask(self, parameters):
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        border_mask = torch.unsqueeze(border_mask,0)
        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']
        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0
        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)
        return border_mask
        
    def swap_core(self, img, kps_5, kps=False, s_e=None, t_e=None, parameters=None, control=None, dfm_model=False):
        s_e = s_e if isinstance(s_e, np.ndarray) else []
        t_e = t_e if isinstance(t_e, np.ndarray) else []
        parameters = parameters or {}
        control = control or {}
        swapper_model = parameters['SwapModelSelection']

        tform = self.get_face_similarity_tform(swapper_model, kps_5)
        original_face_512, original_face_384, original_face_256, original_face_128 = self.get_transformed_and_scaled_faces(tform, img)
        original_faces = (original_face_512, original_face_384, original_face_256, original_face_128)
        dim=1
        
        if (s_e is not None and len(s_e) > 0) or (swapper_model == 'DeepFaceLive (DFM)' and dfm_model):
            input_face_affined, dfm_model, dim, latent = self.get_affined_face_dim_and_swapping_latents(
                original_faces, swapper_model, dfm_model, s_e, t_e, parameters
            )

            if parameters['FaceAdjEnableToggle']:
                input_face_affined = v2.functional.affine(
                    input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleAmountSlider'] / 100, 0, 
                    center=(dim*128/2, dim*128/2), interpolation=v2.InterpolationMode.BILINEAR
                )

            itex = 1
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)

            output_size = int(128 * dim)
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
            input_face_affined = input_face_affined.permute(1, 2, 0)
            input_face_affined = torch.div(input_face_affined, 255.0)

            swap, prev_face = self.get_swapped_and_prev_face(
                output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters
            )
        else:
            swap = original_face_512
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)
                prev_face = torch.div(swap, 255.)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters['StrengthEnableToggle']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100)*0.01
                if alpha==0:
                    alpha=1
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                prev_face = t512(prev_face)
                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)

        border_mask = self.get_border_mask(parameters)
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        swap_mask = torch.unsqueeze(swap_mask,0)
        
        if parameters['FaceExpressionEnableToggle']:
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)

        if parameters["FaceRestorerEnableToggle"]:
            swap = self.models_processor.apply_facerestorer(
                swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], 
                parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], 
                control['DetectorScoreSlider']
            )

        if parameters["FaceRestorerEnable2Toggle"]:
            swap = self.models_processor.apply_facerestorer(
                swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], 
                parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], 
                control['DetectorScoreSlider']
            )

        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256, parameters["OccluderSizeSlider"])
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["DFLXSegEnableToggle"]:
            img_mask = self.models_processor.apply_dfl_xseg(original_face_256, -parameters["DFLXSegSizeSlider"])
            img_mask = t128(img_mask)
            swap_mask = torch.mul(swap_mask, 1 - img_mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["FaceParserEnableToggle"]:
            mask = self.models_processor.apply_face_parser(swap, parameters)
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        if parameters["ClipEnableToggle"]:
            mask = self.models_processor.run_CLIPs(original_face_512, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask = t128(mask)
            swap_mask *= mask

        if parameters['RestoreMouthEnableToggle'] or parameters['RestoreEyesEnableToggle']:
            M = tform.params[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            homogeneous_kps = np.hstack([kps_5, ones_column])
            dst_kps_5 = np.dot(homogeneous_kps, M.T)

            img_swap_mask = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
            img_orig_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()

            if parameters['RestoreMouthEnableToggle']:
                img_swap_mask = self.models_processor.restore_mouth(
                    img_orig_mask, img_swap_mask, dst_kps_5, 
                    parameters['RestoreMouthBlendAmountSlider']/100, 
                    parameters['RestoreMouthFeatherBlendSlider'], 
                    parameters['RestoreMouthSizeFactorSlider']/100, 
                    parameters['RestoreXMouthRadiusFactorDecimalSlider'], 
                    parameters['RestoreYMouthRadiusFactorDecimalSlider'], 
                    parameters['RestoreXMouthOffsetSlider'], 
                    parameters['RestoreYMouthOffsetSlider']
                )
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            if parameters['RestoreEyesEnableToggle']:
                img_swap_mask = self.models_processor.restore_eyes(
                    img_orig_mask, img_swap_mask, dst_kps_5, 
                    parameters['RestoreEyesBlendAmountSlider']/100, 
                    parameters['RestoreEyesFeatherBlendSlider'], 
                    parameters['RestoreEyesSizeFactorDecimalSlider'],  
                    parameters['RestoreXEyesRadiusFactorDecimalSlider'], 
                    parameters['RestoreYEyesRadiusFactorDecimalSlider'], 
                    parameters['RestoreXEyesOffsetSlider'], 
                    parameters['RestoreYEyesOffsetSlider'], 
                    parameters['RestoreEyesSpacingOffsetSlider']
                )
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            gauss = transforms.GaussianBlur(parameters['RestoreEyesMouthBlurSlider']*2+1, (parameters['RestoreEyesMouthBlurSlider']+1)*0.2)
            img_swap_mask = gauss(img_swap_mask)
            img_swap_mask = t128(img_swap_mask)
            swap_mask = torch.mul(swap_mask, img_swap_mask)

        if parameters["DifferencingEnableToggle"]:
            mask = self.models_processor.apply_fake_diff(swap, original_face_512, parameters["DifferencingAmountSlider"])
            gauss = transforms.GaussianBlur(parameters['DifferencingBlendAmountSlider']*2+1, (parameters['DifferencingBlendAmountSlider']+1)*0.2)
            mask = gauss(mask.type(torch.float32))
            swap = swap * mask + original_face_512*(1-mask)

        if parameters["AutoColorEnableToggle"]:
            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])
            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, t512(swap_mask), parameters["AutoColorBlendAmountSlider"])
            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])
            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, t512(swap_mask), parameters["AutoColorBlendAmountSlider"])

        if parameters['ColorEnableToggle']:
            swap = torch.unsqueeze(swap,0).contiguous()
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaDecimalSlider'], 1.0)
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device)
            swap += del_color
            swap = torch.clamp(swap, min=0., max=255.)
            swap = swap.permute(2, 0, 1).type(torch.uint8)

            swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightnessDecimalSlider'])
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastDecimalSlider'])
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationDecimalSlider'])
            swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessDecimalSlider'])
            swap = v2.functional.adjust_hue(swap, parameters['ColorHueDecimalSlider'])

            if parameters['ColorNoiseDecimalSlider'] > 0:
                swap = swap.permute(1, 2, 0).type(torch.float32)
                swap = swap + parameters['ColorNoiseDecimalSlider']*torch.randn(512, 512, 3, device=self.models_processor.device)
                swap = torch.clamp(swap, 0, 255)
                swap = swap.permute(2, 0, 1)

        if parameters['JPEGCompressionEnableToggle']:
            try:
                swap = faceutil.jpegBlur(swap, parameters["JPEGCompressionAmountSlider"])
            except:
                pass
                
        if parameters['FinalBlendAdjEnableToggle'] and parameters['FinalBlendAdjEnableToggle'] > 0:
            final_blur_strength = parameters['FinalBlendAmountSlider']
            kernel_size = 2 * final_blur_strength + 1
            sigma = final_blur_strength * 0.1
            gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

        gauss = transforms.GaussianBlur(parameters['OverallMaskBlendAmountSlider'] * 2 + 1, (parameters['OverallMaskBlendAmountSlider'] + 1) * 0.2)
        swap_mask = gauss(swap_mask)
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)
        swap = torch.mul(swap, swap_mask)

        original_face_512_clone = None
        if self.is_view_face_compare:
            original_face_512_clone = original_face_512.clone()
            original_face_512_clone = original_face_512_clone.type(torch.uint8)
            original_face_512_clone = original_face_512_clone.permute(1, 2, 0)
            
        swap_mask_clone = None
        if self.is_view_face_mask:
            swap_mask_clone = swap_mask.clone()
            swap_mask_clone = torch.sub(1, swap_mask_clone)
            swap_mask_clone = torch.cat((swap_mask_clone,swap_mask_clone,swap_mask_clone),0)
            swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
            swap_mask_clone = torch.mul(swap_mask_clone, 255.).type(torch.uint8)

        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])
        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])

        left = floor(np.min(x))
        if left <0: left=0
        top = floor(np.min(y))
        if top <0: top=0
        right = ceil(np.max(x))
        if right >img.shape[2]: right=img.shape[2]
        bottom = ceil(np.max(y))
        if bottom >img.shape[1]: bottom=img.shape[1]

        swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap = v2.functional.affine(
            swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), 
            tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0)
        )
        swap = swap[0:3, top:bottom, left:right]
        swap = swap.permute(1, 2, 0)

        swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
        swap_mask = v2.functional.affine(
            swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), 
            tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center=(0,0)
        )
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask = swap_mask.permute(1, 2, 0)
        swap_mask = torch.sub(1, swap_mask)

        img_crop = img[0:3, top:bottom, left:right]
        img_crop = img_crop.permute(1,2,0)
        img_crop = torch.mul(swap_mask,img_crop)
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.permute(2,0,1)
        img[0:3, top:bottom, left:right] = swap

        return img, original_face_512_clone, swap_mask_clone

    def enhance_core(self, img, control):
        enhancer_type = control['FrameEnhancerTypeSelection']
        match enhancer_type:
            case 'RealEsrgan-x2-Plus' | 'RealEsrgan-x4-Plus' | 'BSRGan-x2' | 'BSRGan-x4' | 'UltraSharp-x4' | 'UltraMix-x4' | 'RealEsr-General-x4v3':
                tile_size = 512
                if enhancer_type == 'RealEsrgan-x2-Plus' or enhancer_type == 'BSRGan-x2':
                    scale = 2
                else:
                    scale = 4
                image = img.type(torch.float32)
                if torch.max(image) > 256:
                    max_range = 65535
                else:
                    max_range = 255
                image = torch.div(image, max_range)
                image = torch.unsqueeze(image, 0).contiguous()
                image = self.models_processor.run_enhance_frame_tile_process(image, enhancer_type, tile_size=tile_size, scale=scale)
                image = torch.squeeze(image)
                image = torch.clamp(image, 0, 1)
                image = torch.mul(image, max_range)
                alpha = float(control["FrameEnhancerBlendSlider"])/100.0
                t_scale = v2.Resize((img.shape[1] * scale, img.shape[2] * scale), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                img = t_scale(img)
                img = torch.add(torch.mul(image, alpha), torch.mul(img, 1-alpha))
                if max_range == 255:
                    img = img.type(torch.uint8)
                else:
                    img = img.type(torch.uint16)

            case 'DeOldify-Artistic' | 'DeOldify-Stable' | 'DeOldify-Video':
                render_factor = 384
                _, h, w = img.shape
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)
                image = image.type(torch.float32)
                image = torch.unsqueeze(image, 0).contiguous()
                output = torch.empty((image.shape), dtype=torch.float32, device=self.models_processor.device).contiguous()
                match enhancer_type:
                    case 'DeOldify-Artistic':
                        self.models_processor.run_deoldify_artistic(image, output)
                    case 'DeOldify-Stable':
                        self.models_processor.run_deoldify_stable(image, output)
                    case 'DeOldify-Video':
                        self.models_processor.run_deoldify_video(image, output)
                output = torch.squeeze(output)
                t_resize_o = v2.Resize((h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output = t_resize_o(output)
                output = faceutil.rgb_to_yuv(output, True)
                hires = faceutil.rgb_to_yuv(img, True)
                hires[1:3, :, :] = output[1:3, :, :]
                hires = faceutil.yuv_to_rgb(hires, True)
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                img = torch.add(torch.mul(hires, alpha), torch.mul(img, 1-alpha))
                img = img.type(torch.uint8)

            case 'DDColor-Artistic' | 'DDColor':
                render_factor = 384
                orig_l = faceutil.rgb_to_lab(img, True)
                orig_l = orig_l[0:1, :, :]
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)
                img_l = faceutil.rgb_to_lab(image, True)
                img_l = img_l[0:1, :, :]
                img_gray_lab = torch.cat((img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)), dim=0)
                img_gray_rgb = faceutil.lab_to_rgb(img_gray_lab)
                tensor_gray_rgb = torch.unsqueeze(img_gray_rgb.type(torch.float32), 0).contiguous()
                output_ab = torch.empty((1, 2, render_factor, render_factor), dtype=torch.float32, device=self.models_processor.device)
                match enhancer_type:
                    case 'DDColor-Artistic':
                        self.models_processor.run_ddcolor_artistic(tensor_gray_rgb, output_ab)
                    case 'DDColor':
                        self.models_processor.run_ddcolor(tensor_gray_rgb, output_ab)
                output_ab = output_ab.squeeze(0)
                t_resize_o = v2.Resize((img.size(1), img.size(2)), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output_lab_resize = t_resize_o(output_ab)
                output_lab = torch.cat((orig_l, output_lab_resize), dim=0)
                output_rgb = faceutil.lab_to_rgb(output_lab, True)
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                blended_img = torch.add(torch.mul(output_rgb, alpha), torch.mul(img, 1 - alpha))
                img = blended_img.type(torch.uint8)
        return img

    def apply_face_expression_restorer(self, driving, target, parameters):
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(
            driving, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False
        )
        driving_face_512 = driving.clone()
        driving_face_256 = t256(driving_face_512)
        c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None])
        c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None])
        x_d_i_info = self.models_processor.lp_motion_extractor(driving_face_256, 'Human-Face')
        R_d_i = faceutil.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
        
        driving_multiplier = parameters['FaceExpressionFriendlyFactorDecimalSlider']
        animation_region = parameters['FaceExpressionAnimationRegionSelection']
        flag_normalize_lip = parameters['FaceExpressionNormalizeLipsEnableToggle']
        lip_normalize_threshold = parameters['FaceExpressionNormalizeLipsThresholdDecimalSlider']
        flag_eye_retargeting = parameters['FaceExpressionRetargetingEyesEnableToggle']
        eye_retargeting_multiplier = parameters['FaceExpressionRetargetingEyesMultiplierDecimalSlider']
        flag_lip_retargeting = parameters['FaceExpressionRetargetingLipsEnableToggle']
        lip_retargeting_multiplier = parameters['FaceExpressionRetargetingLipsMultiplierDecimalSlider']
        
        if animation_region == 'all':
            animation_region = 'eyes,lips'

        flag_relative_motion = True
        flag_stitching = True
        flag_pasteback = True
        flag_do_crop = True
        
        lip_delta_before_animation, eye_delta_before_animation = None, None
        target = torch.clamp(target, 0, 255).type(torch.uint8)
        _, source_lmk, _ = self.models_processor.run_detect_landmark(
            target, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False
        )
        target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
            target, source_lmk, dsize=512, scale=parameters['FaceExpressionCropScaleDecimalSlider'], 
            vy_ratio=parameters['FaceExpressionVYRatioDecimalSlider'], interpolation=v2.InterpolationMode.BILINEAR
        )
        target_face_256 = t256(target_face_512)
        x_s_info = self.models_processor.lp_motion_extractor(target_face_256, 'Human-Face')
        x_c_s = x_s_info['kp']
        R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.models_processor.lp_appearance_feature_extractor(target_face_256, 'Human-Face')
        x_s = faceutil.transform_keypoint(x_s_info)

        if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = faceutil.calc_combined_lip_ratio(
                c_d_lip_before_animation, source_lmk, device=self.models_processor.device
            )
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.models_processor.lp_retarget_lip(
                    x_s, combined_lip_ratio_tensor_before_animation
                )

        delta_new = x_s_info['exp'].clone()
        if flag_relative_motion:
            if animation_region == "all" or animation_region == "pose":
                R_new = (R_d_i @ R_d_i.permute(0, 2, 1)) @ R_s
            else:
                R_new = R_s
            if animation_region == "all" or animation_region == "exp":
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device))
            else:
                if "lips" in animation_region:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, lip_idx, :]
                if "eyes" in animation_region:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            if animation_region == "all":
                scale_new = x_s_info['scale']
            else:
                scale_new = x_s_info['scale']
            if animation_region == "all" or animation_region == "pose":
                t_new = x_s_info['t']
            else:
                t_new = x_s_info['t']
        else:
            if animation_region == "all" or animation_region == "pose":
                R_new = R_d_i
            else:
                R_new = R_s
            if animation_region == "all" or animation_region == "exp":
                for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            else:
                if "lips" in animation_region:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
                if "eyes" in animation_region:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            scale_new = x_s_info['scale'] 
            if animation_region == "all" or animation_region == "pose":
                t_new = x_d_i_info['t']
            else:
                t_new = x_s_info['t']

        t_new[..., 2].fill_(0)
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
        
        if not flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new += lip_delta_before_animation
        elif flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"]) + lip_delta_before_animation
            else:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])
        else:
            eyes_delta, lip_delta = None, None
            if flag_eye_retargeting and source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device=self.models_processor.device)
                combined_eye_ratio_tensor = combined_eye_ratio_tensor * eye_retargeting_multiplier
                eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])
            if flag_lip_retargeting and source_lmk is not None:
                c_d_lip_i = c_d_lip_lst
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(c_d_lip_i, source_lmk, device=self.models_processor.device)
                combined_lip_ratio_tensor = combined_lip_ratio_tensor * lip_retargeting_multiplier
                lip_delta = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])
            if flag_relative_motion:
                x_d_i_new = x_s + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
            else:
                x_d_i_new = x_d_i_new + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
            if flag_stitching:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])

        x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
        out = self.models_processor.lp_warp_decode(f_s, x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])
        out = torch.squeeze(out)
        out = torch.clamp(out, 0, 1)

        if flag_pasteback and flag_do_crop and flag_stitching:
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (target.shape[1], target.shape[2])
            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(
                out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), 
                scale=t.scale, shear=(0.0, 0.0), interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0)
            )
            out = v2.functional.crop(out, 0,0, dsize[0], dsize[1])

        out = torch.clamp(torch.mul(out, 255.0), 0, 255).type(torch.float32)
        return out

    def swap_edit_face_core(self, img, kps, parameters, control, **kwargs):
        if parameters['FaceEditorEnableToggle']:
            t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
            init_source_eye_ratio = 0.0
            init_source_lip_ratio = 0.0
            _, lmk_crop, _ = self.models_processor.run_detect_landmark(
                img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True
            )
            source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
            source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
            init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                img, lmk_crop, dsize=512, scale=parameters["FaceEditorCropScaleDecimalSlider"], 
                vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=v2.InterpolationMode.BILINEAR
            )
            original_face_256 = t256(original_face_512)
            x_s_info = self.models_processor.lp_motion_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_d_info_user_pitch = x_s_info['pitch'] + parameters['HeadPitchSlider']
            x_d_info_user_yaw = x_s_info['yaw'] + parameters['HeadYawSlider']
            x_d_info_user_roll = x_s_info['roll'] + parameters['HeadRollSlider']
            R_s_user = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = faceutil.get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            f_s_user = self.models_processor.lp_appearance_feature_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_s_user = faceutil.transform_keypoint(x_s_info)
            mov_x = torch.tensor(parameters['XAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_y = torch.tensor(parameters['YAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_z = torch.tensor(parameters['ZAxisMovementDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_x = torch.tensor(parameters['EyeGazeHorizontalDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_y = torch.tensor(parameters['EyeGazeVerticalDecimalSlider']).to(self.models_processor.device)
            smile = torch.tensor(parameters['MouthSmileDecimalSlider']).to(self.models_processor.device)
            wink = torch.tensor(parameters['EyeWinkDecimalSlider']).to(self.models_processor.device)
            eyebrow = torch.tensor(parameters['EyeBrowsDirectionDecimalSlider']).to(self.models_processor.device)
            lip_variation_zero = torch.tensor(parameters['MouthPoutingDecimalSlider']).to(self.models_processor.device)
            lip_variation_one = torch.tensor(parameters['MouthPursingDecimalSlider']).to(self.models_processor.device)
            lip_variation_two = torch.tensor(parameters['MouthGrinDecimalSlider']).to(self.models_processor.device)
            lip_variation_three = torch.tensor(parameters['LipsCloseOpenSlider']).to(self.models_processor.device)
            x_c_s = x_s_info['kp']
            delta_new = x_s_info['exp']
            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                delta_new = faceutil.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
            if smile != 0:
                delta_new = faceutil.update_delta_new_smile(smile, delta_new)
            if wink != 0:
                delta_new = faceutil.update_delta_new_wink(wink, delta_new)
            if eyebrow != 0:
                delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
            if lip_variation_zero != 0:
                delta_new = faceutil.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
            if lip_variation_one != 0:
                delta_new = faceutil.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
            if lip_variation_two != 0:
                delta_new = faceutil.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
            if lip_variation_three != 0:
                delta_new = faceutil.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
            if mov_x != 0:
                delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y != 0:
                delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None

            input_eye_ratio = max(min(init_source_eye_ratio + parameters['EyesOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_eye_ratio != init_source_eye_ratio:
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(
                    [[float(input_eye_ratio)]], lmk_crop, device=self.models_processor.device
                )
                eyes_delta = self.models_processor.lp_retarget_eye(x_s_user, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            input_lip_ratio = max(min(init_source_lip_ratio + parameters['LipsOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_lip_ratio != init_source_lip_ratio:
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(
                    [[float(input_lip_ratio)]], lmk_crop, device=self.models_processor.device
                )
                lip_delta = self.models_processor.lp_retarget_lip(x_s_user, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            x_d_new = x_d_new + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)

            flag_stitching_retargeting_input = kwargs.get('flag_stitching_retargeting_input', True)
            if flag_stitching_retargeting_input:
                x_d_new = self.models_processor.lp_stitching(x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])

            out = self.models_processor.lp_warp_decode(f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])
            out = torch.squeeze(out)
            out = torch.clamp(out, 0, 1)

            flag_do_crop_input_retargeting_image = kwargs.get('flag_do_crop_input_retargeting_image', True)
            if flag_do_crop_input_retargeting_image:
                gauss = transforms.GaussianBlur(parameters['FaceEditorBlurAmountSlider']*2+1, (parameters['FaceEditorBlurAmountSlider']+1)*0.2)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)
            else:
                img = out                
                img = torch.mul(img, 255.0)
                img = torch.clamp(img, 0, 255).type(torch.uint8)

        if parameters['FaceMakeupEnableToggle'] or parameters['HairMakeupEnableToggle'] or parameters['EyeBrowsMakeupEnableToggle'] or parameters['LipsMakeupEnableToggle']:
            _, lmk_crop, _ = self.models_processor.run_detect_landmark(
                img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True
            )
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
                img, lmk_crop, dsize=512, scale=parameters['FaceEditorCropScaleDecimalSlider'], 
                vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=v2.InterpolationMode.BILINEAR
            )
            out, mask_out = self.models_processor.apply_face_makeup(original_face_512, parameters)
            if 1:
                gauss = transforms.GaussianBlur(5*2 +1, (5+1)*0.2)
                out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img