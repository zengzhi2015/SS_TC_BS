#include "BACKSUB.h"

int main()
{
    String dataset_logits_Path("/home/Datasets/CDnet2014_SS_Logits/");
    String raw_bin_root("/home/Datasets/SWCD_results/raw_bin/");
    String SSA_bin_root("/home/Datasets/SWCD_results/SSA_bin/");
    vector<String> CATAGORY(53);
    CATAGORY[0] = "baseline/highway/";
    CATAGORY[1] = "baseline/office/";
    CATAGORY[2] = "baseline/pedestrians/";
    CATAGORY[3] = "baseline/PETS2006/";
    CATAGORY[4] = "cameraJitter/badminton/";
    CATAGORY[5] = "cameraJitter/boulevard/";
    CATAGORY[6] = "cameraJitter/sidewalk/";
    CATAGORY[7] = "cameraJitter/traffic/";
    CATAGORY[8] = "dynamicBackground/boats/";
    CATAGORY[9] = "dynamicBackground/canoe/";
    CATAGORY[10] = "dynamicBackground/fall/";
    CATAGORY[11] = "dynamicBackground/fountain01/";
    CATAGORY[12] = "dynamicBackground/fountain02/";
    CATAGORY[13] = "dynamicBackground/overpass/";
    CATAGORY[14] = "intermittentObjectMotion/abandonedBox/";
    CATAGORY[15] = "intermittentObjectMotion/parking/";
    CATAGORY[16] = "intermittentObjectMotion/sofa/";
    CATAGORY[17] = "intermittentObjectMotion/streetLight/";
    CATAGORY[18] = "intermittentObjectMotion/tramstop/";
    CATAGORY[19] = "intermittentObjectMotion/winterDriveway/";
    CATAGORY[20] = "shadow/backdoor/";
    CATAGORY[21] = "shadow/bungalows/";
    CATAGORY[22] = "shadow/busStation/";
    CATAGORY[23] = "shadow/copyMachine/";
    CATAGORY[24] = "shadow/cubicle/";
    CATAGORY[25] = "shadow/peopleInShade/";
    CATAGORY[26] = "thermal/corridor/";
    CATAGORY[27] = "thermal/diningRoom/";
    CATAGORY[28] = "thermal/lakeSide/";
    CATAGORY[29] = "thermal/library/";
    CATAGORY[30] = "thermal/park/";
    CATAGORY[31] = "badWeather/blizzard/";
    CATAGORY[32] = "badWeather/skating/";
    CATAGORY[33] = "badWeather/snowFall/";
    CATAGORY[34] = "badWeather/wetSnow/";
    CATAGORY[35] = "lowFramerate/port_0_17fps/";
    CATAGORY[36] = "lowFramerate/tramCrossroad_1fps/";
    CATAGORY[37] = "lowFramerate/tunnelExit_0_35fps/";
    CATAGORY[38] = "lowFramerate/turnpike_0_5fps/";
    CATAGORY[39] = "nightVideos/bridgeEntry/";
    CATAGORY[40] = "nightVideos/busyBoulvard/";
    CATAGORY[41] = "nightVideos/fluidHighway/";
    CATAGORY[42] = "nightVideos/streetCornerAtNight/";
    CATAGORY[43] = "nightVideos/tramStation/";
    CATAGORY[44] = "nightVideos/winterStreet/";
    CATAGORY[45] = "PTZ/continuousPan/";
    CATAGORY[46] = "PTZ/intermittentPan/";
    CATAGORY[47] = "PTZ/twoPositionPTZCam/";
    CATAGORY[48] = "PTZ/zoomInZoomOut/";
    CATAGORY[49] = "turbulence/turbulence0/";
    CATAGORY[50] = "turbulence/turbulence1/";
    CATAGORY[51] = "turbulence/turbulence2/";
    CATAGORY[52] = "turbulence/turbulence3/";



    int CT_num_init = 0;
    int CT_num_end = 52;
    for(int CT_num = CT_num_init; CT_num <= CT_num_end; CT_num++) {
        String frame_logits_path = dataset_logits_Path + CATAGORY[CT_num];
        String raw_bin_dir = raw_bin_root + CATAGORY[CT_num];
        String SSA_bin_dir = SSA_bin_root + CATAGORY[CT_num];

        BackgroundSubtraction(frame_logits_path,
                              raw_bin_dir,
                              SSA_bin_dir);
    }

    return 0;
}


