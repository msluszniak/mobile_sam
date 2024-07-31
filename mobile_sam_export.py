import torch
#import logging
#from torch._export import capture_pre_autograd_graph

from torch.export import export, ExportedProgram
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
# from executorch.backends.apple.coreml.compiler import CoreMLBackend
# from executorch.backends.apple.coreml.compiler import CoreMLBackend
# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location("__init__", "/Users/mateuszsluszniak/Desktop/ai/sam_research/executorch_3/executorch/backends/apple/coreml/compiler/__init__.py")
# foo = importlib.util.module_from_spec(spec)
# sys.modules["__init__"] = foo
# spec.loader.exec_module(foo)
# # foo.MyClass()
import cv2
import numpy as np
#from segment_anything import sam_model_registry  # @manual
from mobile_sam import sam_model_registry 
# from mobile_sam import ImageEncoderViT, TinyViT, PromptEncoder
from executorch import exir
from executorch.exir.passes import MemoryPlanningPass
# import matplotlib.pyplot as plt
import copy
# from executorch.sdk import generate_etrecord
#import os
#print(os.getcwd())

#import pkgutil
#import executorch.examples
#print(list(pkgutil.iter_modules(executorch.examples.__path__)))
import sys
sys.path.append('..')
#from executorch_1.executorch.extension.export_util.utils import export_to_exec_prog, save_pte_program
from executorch.examples.portable.scripts.export import export_to_exec_prog, save_pte_program
#from executorch.examples
#from executorch.examples.models import *

class SegmentAnythingModel:
    def __init__(self):
        # Use Tiny
        self.model_type = "vit_t"

    def get_eager_model(self) -> torch.nn.Module:
        #logging.info(f"Loading segment-anything {self.model_type} model")
        self.sam_model = sam_model_registry[self.model_type]("/Users/mateuszsluszniak/Desktop/ai/sam_research/new_mobile_sam/MobileSAM/weights/mobile_sam.pt")
        #logging.info(f"Loaded segment-anything {self.model_type} model")
        return self.sam_model

    def get_example_inputs(self):
        #embed_size = self.sam_model.prompt_encoder.image_embedding_size
        #mask_input_size = [4 * x for x in embed_size]
        #print(mask_input_size)
        IMAGE_PATH = './yin_yang.png'
        im = cv2.imread(IMAGE_PATH)
        img0 = im.copy()
        im = cv2.resize(im, (1024, 1024), interpolation = cv2.INTER_AREA)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        # Convert into torch
        im = torch.from_numpy(im)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        #test
        # im = torch.ones((3, 1024, 1024), dtype=torch.float32)
        batched_input = [
            # NOTE: SAM can take any of the following inputs independently. For
            # example, if you want to gen an inference model with point-only inputs,
            # just comment out the other inputs.
            {  # multi-points input
               "image": im,
               "original_size": (1024, 1024),
               "point_coords": torch.Tensor([[[222, 444], [333, 555]]]),
               "point_labels": torch.Tensor([[1, 1]]),
            },
            #{  # multi-boxes input
            #    "image": torch.randn(3, 224, 224),
            #    "original_size": (1500, 2250),
            #    "boxes": torch.randn(2, 4),
            #},
            # {  # mask input
            #     # "image": torch.randn(3, 1000, 1000),
            #     "image": im,
            #     "original_size": (1024, 1024),
            #     "mask_inputs": torch.randn(1, 1, *[1024, 1024]),
            # },
            #{  # comb input
            #    "image": torch.randn(3, 224, 224),
            #    "original_size": (1500, 2250),
            #    "point_coords": torch.randint(low=0, high=224, size=(3, 5, 2)),
            #    "point_labels": torch.randint(low=0, high=4, size=(3, 5)),
            #    "boxes": torch.randn(3, 4),
            #    "mask_input": torch.randn(3, 1, *mask_input_size),
            #},
        ]
        multimask_output = True
        return (batched_input, multimask_output)
        # return batched_input
    

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

with torch.no_grad():
    model = SegmentAnythingModel()
    og_model = model.get_eager_model()
    inputs = model.get_example_inputs()
    model_inputs = (inputs[0][0]['image'].unsqueeze(0), inputs[0][0]['point_coords'], inputs[0][0]['point_labels'])
    # print(inputs[0][0]['point_coords'].shape)
    og_model.eval()

    model_name = "mobile_sam_vit"
    #print(og_model(*inputs))
    
    #m = capture_pre_autograd_graph(og_model, (inputs,))


    # prompt_embed_dim = 256
    # image_size = 1024
    # vit_patch_size = 16
    # image_embedding_size = image_size // vit_patch_size
    # og_model = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
    #             embed_dims=[64, 128, 160, 320],
    #             depths=[2, 2, 6, 2],
    #             num_heads=[2, 4, 5, 10],
    #             window_sizes=[7, 7, 14, 7],
    #             mlp_ratio=4.,
    #             drop_rate=0.,
    #             drop_path_rate=0.0,
    #             use_checkpoint=False,
    #             mbconv_expand_ratio=4.0,
    #             local_conv_size=3,
    #             layer_lr_decay=0.8
    #         )

   #prompt_embed_dim = 256
   #image_size = 1024
   # image_size = 2048
   #vit_patch_size = 16
    #image_embedding_size = image_size // vit_patch_size
    #og_model = PromptEncoder(
     #       embed_dim=prompt_embed_dim,
     #       image_embedding_size=(image_embedding_size, image_embedding_size),
     #       input_image_size=(image_size, image_size),
     #       mask_in_chans=16,
     #       )
    #og_model.eval()
    #prog = export(og_model, (None, None, None))

    # print(og_model)
    # # output = og_model(model_inputs)
    # # output = output[0]
    # # masks = output["masks"]
    # # scores = output["iou_predictions"]
    # # logits = output["low_res_logits"]

    # # masks = masks[0]
    # # scores = scores[0]
    # # logits = logits[0]
    # masks = og_model(model_inputs)
    # # print(masks.shape, scores, logits.shape)
    # print(masks.shape)
    # for i, mask in enumerate(masks):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(inputs[0][0]["image"].permute(1, 2, 0))
    #     show_mask(mask, plt.gca())
    #     show_points(inputs[0][0]["point_coords"], inputs[0][0]["point_labels"], plt.gca())
    #     # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()
    #     plt.show() 
    prog = export(og_model, model_inputs)
    # prog = export(og_model, tuple())
    edge = to_edge(prog, compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),)
    print(edge.exported_program())
    edge = edge.to_backend(XnnpackPartitioner())
    # edge.to_backend(#foo.CoreMLBackend.__name__,
    #   edge.exported_program())
    # edge = edge.to_backend()
    # exec_prog = edge.to_executorch()
    exec_prog = edge.to_executorch(
        #         config=exir.ExecutorchBackendConfig(
        #     extract_constant_segment=False, extract_delegate_segments=True
        # )
            exir.ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    memory_planning_algo="greedy",
                    alloc_graph_input=False, # Inputs will not be memory planned, the data_ptr for input tensors after model load will be nullptr
                    alloc_graph_output=True, # Outputs will be memory planned, the data_ptr for input tensors after model load will be in the `planned_memory`.
                )
            )
        )

    # etrecord_path = "etrecord.bin"
    # generate_etrecord(etrecord_path, edge_manager_copy, exec_prog)

    with open("mobile_sam_app.pte", "wb") as file:
        exec_prog.write_to_file(file)

    # prog = export_to_exec_prog(og_model, inputs, edge_compile_config=EdgeCompileConfig(_check_ir_validity=False), backend_config = XnnpackPartitioner(),)
    # prog.to_backend()
    # save_pte_program(prog, model_name) 

    # pre_autograd_aten_dialect = capture_pre_autograd_graph(og_model, inputs)
    # aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, inputs)
    # edge_program: EdgeProgramManager = to_edge(aten_dialect)
    # edge_program: EdgeProgramManager = edge_program.to_backend(XnnpackPartitioner)
    # executorch_program: ExecutorchProgramManager = edge_program.to_executorch(ExecutorchBackendConfig(passes=[]))
    # with open('mobile_sam_xnnpack.pte', 'wb') as file:
    #     file.write(prog.buffer)


