#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <filesystem>
#include <regex>
#include "face2embng.h"
#include <nlohmann/json.hpp>

using namespace std;

vector<char> ReadFile(const string& filename)
{
    ifstream ifs(filename, ios::binary|ios::ate);
    ifstream::pos_type pos = ifs.tellg();

    std::vector<char> result(pos);

    ifs.seekg(0, ios::beg);
    ifs.read(result.data(), pos);

    return result;
}
template<typename T_stream_type, typename ...Args>
std::unique_ptr<T_stream_type> OpenFileOrDie(const std::string& file, const Args&... args)
{
    std::unique_ptr<T_stream_type> stream_ptr(new T_stream_type(file, args...));
    if(stream_ptr->is_open())
        return stream_ptr;
    else
        throw std::runtime_error("Can't open file " + file);
}

inline nlohmann::json ParseJsonItem(const std::string& path)
{
    nlohmann::json config;
    auto stream_ptr = OpenFileOrDie<std::ifstream>(path);
    try
    {
        *stream_ptr >> config;
        return config;
    }
    catch(const nlohmann::detail::exception& err)
    {
        throw std::runtime_error("error parse JSON item " + path + ": " + err.what());
    }
}

void get_floats4json_str( std::string& str_list, std::vector<float>& extracted_floats )
{
    extracted_floats.clear();

    std::regex  float_regex(R"([-\d\.e\+]+)");
    std::smatch match;

    // Iterate through the input string and extract floats
    while( std::regex_search(str_list, match, float_regex) ) {
        extracted_floats.push_back( std::stof( match[0].str() ) );
        // Update the input string to skip the matched portion
        str_list = match.suffix().str();
    }
}
void get_faces4json( const std::string& in_file, std::vector<face_info>&  faces )
{
    nlohmann::json in_json = ParseJsonItem( in_file );
    // parse faces
    if( in_json.find("tags_list") == in_json.end())  return;
    nlohmann::json tags = in_json.at("tags_list");
    std::string ss_temp;

    for (const auto& tag : tags.items() )
    {
        nlohmann::json object = tag.value();
        if( object.find("category") == object.end() ) continue;
        if( object.at("category") != "Faces" ) continue;
        if( object.find("items") == object.end() ) continue;
        //std::cout << object.dump() << std::endl;
        nlohmann::json items = object.at("items");
        int cur_id = 0;
        for (const auto& item : items.items() )
        {
            nlohmann::json item_obj = item.value();
            //std::cout << item_obj.dump() << std::endl;
            if( item_obj.find("id") != item_obj.end() )
                cur_id = item_obj.at("id").get<int>(); // id from json
            else
                cur_id++;                               // id <- item index
            if( item_obj.find("bbox") == item_obj.end() || item_obj.find("embedding") == item_obj.end() ) continue;
            face_info tmp_face( in_file, cur_id );
            ss_temp = item_obj.at("bbox");
            //std::cout << "bbox:" << ss_temp << std::endl;
            std::vector<float> extracted_floats;
            get_floats4json_str(ss_temp, extracted_floats );

            if( extracted_floats.size() == 4 )
            {
                tmp_face.bbox.x = extracted_floats[0];
                tmp_face.bbox.y = extracted_floats[1];
                tmp_face.bbox.width = extracted_floats[2] - extracted_floats[0];
                tmp_face.bbox.height = extracted_floats[3] - extracted_floats[1];
            }

            ss_temp = item_obj.at("embedding");
            //std::cout << "embedding:" << ss_temp << std::endl;
            get_floats4json_str(ss_temp, extracted_floats );
            //std::cout << "embedding count:" << extracted_floats.size() << std::endl;
            tmp_face.v_embedding.resize( extracted_floats.size() * sizeof(float) );
            memcpy(tmp_face.v_embedding.data(), extracted_floats.data(), tmp_face.v_embedding.size() );

            faces.push_back( tmp_face );
        }
    }
}

float get_rc_int_percent( const cv::Rect_<float>& bbox1, const cv::Rect_<float>& bbox2 )
{
    float xa1 = bbox1.x, xa2 = bbox1.x + bbox1.width, ya1 = bbox1.y, ya2 = bbox1.y + bbox1.height;
    float xb1 = bbox2.x, xb2 = bbox2.x + bbox2.width, yb1 = bbox2.y, yb2 = bbox2.y + bbox2.height;
    float iLeft = std::max( xa1, xb1 );
    float iRight = std::min( xa2, xb2 );
    float iTop = std::max( ya1, yb1 );
    float iBottom = std::min( ya2, yb2 );

    float si = std::max(0.0f, iRight - iLeft) * std::max(0.0f, iBottom - iTop);
    float sa = (xa2 - xa1) * (ya2 - ya1);
    float sb = (xb2 - xb1) * (yb2 - yb1);

    float divider = sa + sb - si;
    return (divider == 0.0f) ? 0.0f : si * 100.0f / divider;
}

void draw_objects(cv::Mat &frame, std::vector<face_info> &Faces, bool draw_text = true)
{
    cv::Scalar color = cv::Scalar(0, 255, 0);

    for(size_t i=0; i < Faces.size(); i++){
        face_info& obj = Faces[i];
        // draw face frame
        cv::rectangle(frame, obj.bbox, color);
        if( draw_text )
        {
            // draw face id
            std::string label = std::to_string( obj.face_id );
            int         baseLine = 0;

            cv::Size label_size = cv::getTextSize( label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine );
            int x = obj.bbox.x;
            int y = obj.bbox.y - label_size.height - baseLine;
            if( y < 0 ) y = 0;
            if( x + label_size.width > frame.cols ) x = frame.cols-label_size.width;
            cv::putText(frame, label, cv::Point(x, y + label_size.height + 2),cv::FONT_HERSHEY_SIMPLEX, 0.6, color);
        }
    }
}

void draw_objects(cv::Mat &frame, std::vector<bbox_info> &Faces)
{
    cv::Scalar color = cv::Scalar(0, 255, 0);

    for(size_t i=0; i < Faces.size(); i++){
        bbox_info& obj = Faces[i];
        // draw face frame
        cv::rectangle(frame, obj.hbox, color);
    }
}

std::string get_bboxes4json( const std::string& in_json, std::vector<bbox_info>& bboxes )
{
    std::string ss_file;

    nlohmann::json json = nlohmann::json::parse(in_json);

    if( json.find("ID") == json.end() || json.find("gtboxes") == json.end() ) return ss_file;

    ss_file = json.at("ID").get<std::string>();
    nlohmann::json gtboxes = json.at("gtboxes");
    int cur_id = 0;
    for (const auto& item : gtboxes.items() )
    {
        nlohmann::json item_obj = item.value();
        if( item_obj.at("tag") != "person" ) continue;
        std::vector<int> box = item_obj.at("hbox").get<std::vector<int>>();
        if( box.size() != 4 ) continue;
        bbox_info new_bbox;
        new_bbox.hbox.x = box[0];
        new_bbox.hbox.y = box[1];
        new_bbox.hbox.width = box[2];
        new_bbox.hbox.height = box[3];

        nlohmann::json head_attr = item_obj.at("head_attr");
        if( head_attr.find("ignore") == head_attr.end() ) continue;
        if( head_attr.at("ignore").get<int>() == 1 ) continue;

        bboxes.push_back( new_bbox );
    }

    return ss_file;
}
int main(int argc, char *argv[])
{
    if( argc < 3 )
    {
        cout << "usage: face_test test_data_path config_path" << endl;
        return 0;
    }

    CFace2Embng             embedder;
    CFromPyEmbng            embedder_py;
    std::string             test_data_path = argv[1], config_path = argv[2], line, file;
    std::vector<face_info>  test_dataset;
    // dataset markup
    std::ifstream           cat_file( test_data_path + "/extract.txt");
    std::ifstream           odgt_file( test_data_path + "/annotation_val.odgt");
    std::ifstream           lbl_file( test_data_path + "/label.txt");

    std::cout << "Test data path:" << test_data_path << std::endl;
    std::cout << "Config path:" << config_path << std::endl;


    // statistics
    uint64_t                n_files = 0;
    uint64_t                TP_cc = 0, TN_cc = 0, FP_cc = 0, FN_cc = 0, n_matches_cc = 0, skip_matches_cc = 0, matches_true_cc = 0, matches_false_cc = 0;
    uint64_t                TP_py = 0, TN_py = 0, FP_py = 0, FN_py = 0, n_matches_py = 0, skip_matches_py = 0, matches_true_py = 0, matches_false_py = 0;
    CFace2Embng::Distance   dist_type = CFace2Embng::EUCL;
    float                   f_bound = (dist_type == CFace2Embng::COS) ? 0.32f : 1.16f;

    if( lbl_file.good() )
    {//WIDER dataset - test detector only bbox
        embedder.initialize( config_path );
        // parse annotation_val.odgt line by line and process json
        std::vector<bbox_info>  vec_bboxes;
        std::string             img_file, json_file;
        bool b_eof = false;
        while( !b_eof )
        {
            std::getline(lbl_file, line);
            //std::cout << "line:" << line << std::endl;
            if( line.size() == 0 ) b_eof = true;
            // ss is an object of stringstream that references the S string.
            std::stringstream           ss(line);
            std::vector<std::string>    vec_tokens;
            std::string                 ss_token;
            // Use while loop to check the getline() function condition.
            while (getline(ss, ss_token, ' '))
                vec_tokens.push_back(ss_token);

            if( (vec_tokens.size() == 2 && vec_tokens[0] == "#") || b_eof )
            { // new file
                if( vec_bboxes.size() > 0 )
                { // process bboxes for file
                    // load image
                    /*cv::Mat frame = cv::imread(img_file, 1), frame_cpy, frame_cpy1;
                    if( frame.empty() ){
                        fprintf(stderr, "cv::imread %s failed\n", img_file.c_str());
                        continue;
                    }
                    frame.copyTo( frame_cpy );
                    frame.copyTo( frame_cpy1 );
                    // draw bboxes
                    draw_objects(frame, vec_bboxes );
                    // draw image
                    cv::imshow("GT bboxes", frame);*/

                    //find faces in image
                    std::vector<face_info>  cc_dataset;
                    embedder.createTemplate(img_file, cc_dataset, true );

                    /*draw_objects(frame_cpy, cc_dataset, false);
                    cv::imshow("From C++", frame_cpy);*/

                    // compare bboxes
                    int true_positives = 0;
                    //int false_positives = 0;
                    int false_negatives = 0;
                    for( const auto& src_bbox : vec_bboxes )
                    {
                        bool b_matched = false;
                        for( auto& new_face : cc_dataset )
                        {
                            float overlapping = get_rc_int_percent( src_bbox.hbox, new_face.bbox );
                            n_matches_cc++;
                            if( overlapping > 50.0f ) // face boxes overlapping more that 50%
                            {// TRUE
                                b_matched = true;
                                break;
                            }
                        }
                        if( b_matched )
                            true_positives++;
                        else
                            false_negatives++;
                    }
                    //false_positives = vec_bboxes.size() - true_positives;
                    TP_cc += true_positives;
                    FN_cc += false_negatives;
                    FP_cc += vec_bboxes.size() - true_positives;

                    // py compare
                    std::vector<face_info>  py_dataset;
                    get_faces4json( json_file, py_dataset );

                    /*draw_objects(frame_cpy1, py_dataset, false);
                    cv::imshow("From py face_detect json", frame_cpy1);

                    char esc = cv::waitKey(10000);
                    if(esc == 27) break;*/

                    true_positives = 0;
                    //int false_positives = 0;
                    false_negatives = 0;
                    for( const auto& src_bbox : vec_bboxes )
                    {
                        bool b_matched = false;
                        for( auto& new_face : py_dataset )
                        {
                            float overlapping = get_rc_int_percent( src_bbox.hbox, new_face.bbox );
                            n_matches_py++;
                            if( overlapping > 50.0f ) // face boxes overlapping more that 50%
                            {// TRUE
                                b_matched = true;
                                break;
                            }
                        }
                        if( b_matched )
                            true_positives++;
                        else
                            false_negatives++;
                    }
                    //false_positives = vec_bboxes.size() - true_positives;
                    TP_py += true_positives;
                    FN_py += false_negatives;
                    FP_py += vec_bboxes.size() - true_positives;

                    vec_bboxes.clear();
                }
                if( vec_tokens.size() == 2 )
                {
                    img_file = test_data_path + "/" + vec_tokens[1];
                    json_file = img_file + ".json";
                    std::cout << "file:" << img_file << std::endl;
                    n_files++;
                }
            }
            if( vec_tokens.size() == 4 )
            {// bbox
                bbox_info tmp_bbox;
                tmp_bbox.hbox.x = std::stoi(vec_tokens[0]);
                tmp_bbox.hbox.y = std::stoi(vec_tokens[1]);
                tmp_bbox.hbox.width = std::stoi(vec_tokens[2]);
                tmp_bbox.hbox.height = std::stoi(vec_tokens[3]);
                vec_bboxes.push_back( tmp_bbox );
                //std::cout << "bbox:" << "x:" << tmp_bbox.hbox.x << " y:" << tmp_bbox.hbox.y << " w:" << tmp_bbox.hbox.width << " h:" << tmp_bbox.hbox.height << std::endl;
            }
        }
    }
    else
    if( odgt_file.good() )
    { // CrowdHuman dataset - test detector only bbox
        embedder.initialize( config_path );
        // parse annotation_val.odgt line by line and process json
        while( std::getline(odgt_file, line) )
        {
            // open json
            std::vector<bbox_info> vec_bboxes;
            std::string id_file = get_bboxes4json( line, vec_bboxes );
            std::string img_file = test_data_path + "/Images/" + id_file + ".jpg";
            std::string json_file = img_file + ".json";
            std::cout << "img file:" << img_file << std::endl;
            std::cout << "json_file:" << json_file;
            n_files++;

            // test
            // load image
            cv::Mat frame = cv::imread(img_file, 1), frame_py;
            if( frame.empty() ){
                fprintf(stderr, "cv::imread %s failed\n", img_file.c_str());
                continue;
            }
            //frame.copyTo( frame_py );
            // draw bboxes
            /*draw_objects(frame, vec_bboxes);
            // draw image
            cv::imshow("GT bboxes", frame);
            char esc = cv::waitKey(10000);
            if(esc == 27) break;*/

            //find faces in image
            std::vector<face_info>  cc_dataset;
            embedder.createTemplate(img_file, cc_dataset);

            std::vector<face_info>  py_dataset;
            get_faces4json( json_file, py_dataset );

            /*draw_objects(frame_py, py_dataset);
            cv::imshow("From JSON(python)", frame_py);

            draw_objects(frame, cc_dataset);
            cv::imshow("From C++", frame);

            char esc = cv::waitKey(10000);
            if(esc == 27) break;*/

            // compare bboxes
            int true_positives = 0;
            //int false_positives = 0;
            int false_negatives = 0;
            for( const auto& src_bbox : vec_bboxes )
            {
                bool b_matched = false;
                for( auto& new_face : cc_dataset )
                {
                    float overlapping = get_rc_int_percent( src_bbox.hbox, new_face.bbox );
                    n_matches_cc++;
                    if( overlapping > 50.0f ) // face boxes overlapping more that 50%
                    {// TRUE
                        b_matched = true;
                        break;
                    }
                }
                if( b_matched )
                    true_positives++;
                else
                    false_negatives++;
            }
            //false_positives = vec_bboxes.size() - true_positives;
            TP_cc += true_positives;
            FN_cc += false_negatives;
            FP_cc += vec_bboxes.size() - true_positives;

            true_positives = 0;
            //int false_positives = 0;
            false_negatives = 0;
            for( const auto& src_bbox : vec_bboxes )
            {
                bool b_matched = false;
                for( auto& new_face : py_dataset )
                {
                    float overlapping = get_rc_int_percent( src_bbox.hbox, new_face.bbox );
                    n_matches_py++;
                    if( overlapping > 50.0f ) // face boxes overlapping more that 50%
                    {// TRUE
                        b_matched = true;
                        break;
                    }
                }
                if( b_matched )
                    true_positives++;
                else
                    false_negatives++;
            }
            //false_positives = vec_bboxes.size() - true_positives;
            TP_py += true_positives;
            FN_py += false_negatives;
            FP_py += vec_bboxes.size() - true_positives;

            std::cout << " - processed" << std::endl;
        }
    }
    else
    if( !cat_file.good() )
    {// test CrowdHuman dataset - test recognizer and compare with python face_detect jsons
        bool b_wait = false;
        embedder.initialize( config_path );
        // find json in test dir
        for( auto const& dir_entry : std::filesystem::recursive_directory_iterator{test_data_path} )
        {
            std::string             ss_in_file = dir_entry.path();
            std::size_t             found;

            test_dataset.clear();

            if( !dir_entry.is_regular_file() || (found = ss_in_file.find(".json")) == std::string::npos ) continue; // not are json
            std::string jpg_file = ss_in_file.substr(0, found);
            //jpg_file += ".jpg";
            std::cout << "json:" << ss_in_file << " image file:" << jpg_file << std::endl;
            // open json
            get_faces4json( ss_in_file, test_dataset );
            // open image
            cv::Mat frame = cv::imread(jpg_file, 1), frame_new;
            if(frame.empty()){
                fprintf(stderr, "cv::imread %s failed\n", jpg_file.c_str());
                return -1;
            }
            frame.copyTo( frame_new ); // copy
            std::cout << "main:: img width:" << frame.cols << " height:" << frame.rows << std::endl;
            // draw gt json with box and id
            if( b_wait )
            {// draw to out image
                draw_objects(frame, test_dataset);
                cv::imshow("From JSON(python)", frame);
            }

            // find faces in image
            std::vector<face_info>  new_dataset;

            embedder.createTemplate(jpg_file, new_dataset);
            // compare embedding and set id
            int n_match_single = 0;
            for( const auto& src_face : test_dataset )
            {
                for( auto& new_face : new_dataset )
                {// compare embedding
                    double similarity = 0.0;
                    if( embedder.matchTemplates( src_face.v_embedding, new_face.v_embedding, similarity, dist_type) < 0 )
                    {
                        std::cout << "matchTemplates failed" << std::endl;
                        return 0;
                    }
                    float overlapping = get_rc_int_percent( src_face.bbox, new_face.bbox );
                    n_matches_cc++;
                    if( overlapping >= 50.0f ) // face boxes overlapping more that 50%
                    {// TRUE
                        matches_true_cc++;
                        if( dist_type == CFace2Embng::EUCL )
                        {
                            if( similarity < f_bound )
                            {//match
                                new_face.face_id = src_face.face_id;
                                n_match_single++;
                                //std::cout << "face id:" << new_face.face_id << " overlapping:" << overlapping << std::endl;
                                TP_cc++;
                            }
                            else
                            {
                                FP_cc++;// not match
                                std::cout << "face id:" << new_face.face_id << " overlapping:" << overlapping  << " similarity:" << similarity << std::endl;
                            }
                        }
                        else
                        {
                            if( similarity > f_bound )
                            {// match
                                new_face.face_id = src_face.face_id;
                                n_match_single++;
                                TP_cc++;
                            }
                            else
                                FP_cc++; // not match
                        }
                    }
                    else
                    {// FALSE
                        matches_false_cc++;
                        if( dist_type == CFace2Embng::COS )
                        {
                            if( similarity < f_bound )
                                TN_cc++;
                            else
                            {
                                FN_cc++;
                                std::cout << "face id:" << new_face.face_id << " overlapping:" << overlapping  << " similarity:" << similarity << std::endl;
                            }
                        }
                        else
                        if( dist_type == CFace2Embng::EUCL )
                        {//
                            if( similarity > f_bound )
                                TN_cc++;
                            else
                                FN_cc++;
                        }
                    }
                }
            }
            if( n_match_single != test_dataset.size() )
            { // finded not all
               // copy files to analis
                std::cout << "test faces count:" << test_dataset.size() << " faces matched:" << n_match_single << std::endl;
                // copy files to /tmp
                if( !b_wait )
                {
                    std::string tmp_str = ss_in_file;
                    int n = tmp_str.rfind('/');
                    tmp_str = tmp_str.erase( 0, n + 1 );
                    std::filesystem::copy( ss_in_file, "/tmp/" + tmp_str);

                    tmp_str = jpg_file;
                    n = tmp_str.rfind('/');
                    tmp_str = tmp_str.erase( 0, n + 1 );
                    std::filesystem::copy( jpg_file, "/tmp/" + tmp_str);
                    // write dump images
                    std::string ss_str = tmp_str;
                    n = ss_str.rfind('.');
                    ss_str.insert( n, "_cc" );
                    draw_objects(frame_new, new_dataset);
                    imwrite( "/tmp/" + ss_str, frame_new);

                    ss_str = tmp_str;
                    n = ss_str.rfind('.');
                    ss_str.insert( n, "_py" );
                    draw_objects(frame, test_dataset);
                    imwrite( "/tmp/" + ss_str, frame);
                }
            }

            if( b_wait )
            {// draw to out image
                draw_objects(frame_new, new_dataset);
                cv::imshow("From image(C++)", frame_new);
            }

            // save dump images to file
//            std::string ss_str = jpg_file;
//            int n = ss_str.rfind('.');

//            ss_str.insert( n, "_py" );
//            std::cout << ss_str << std::endl;
//            imwrite( ss_str, frame);

//            ss_str = jpg_file;
//            ss_str.insert( n, "_cc" );
//            std::cout << ss_str << std::endl;
//            imwrite( ss_str, frame_new);

            if( b_wait )
                cv::waitKey(0);
        }
    }
    else
    { // facemetric dataset 18k images, 15 images per person
        #ifdef USE_RETINA
        embedder.initialize( config_path, 320, 320 );
        #else
        embedder.initialize( config_path, 640, 640 );
        #endif
        //embedder.initialize( config_path, 512, 512 );
        // test face metric dataset
        std::string     delimiter = " ";
        size_t          delim_len = delimiter.length();
        //bool            b_from_json = true;

        // parse extract.txt line by line and process test files
        while( std::getline(cat_file, line) )
        {
            //std::cout << line << std::endl;
            size_t pos_start = 0, pos_end, n_id = 0;

            if( (pos_end = line.find(delimiter, pos_start)) != std::string::npos)
            {
                file = line.substr(pos_start, pos_end - pos_start);
                //std::cout << "file:" << file << " ";
                pos_start = pos_end + delim_len;
                if( pos_start < line.size() )
                {
                    n_id = std::stoi( line.substr( pos_start ) );
                    //std::cout << "id:"  << n_id << std::endl;
                    // calc and save embedding
                    test_dataset.push_back( {file, static_cast<int>(n_id)} );
                }
                else
                {
                    std::cout << "unknown cat file format!"  << std::endl;
                    return 0;
                }
            }
        }
        //calc embedding
        size_t  test_dataset_size = test_dataset.size();
        size_t  n_counter = 0, percent_5 = test_dataset_size / 20;
        bool    b_synchro = false;
        size_t  mis_by_py = 0, mis_by_cc = 0;
        for( auto& item : test_dataset )
        {
            embedder_py.createTemplate( test_data_path + "/images/" + item.img_file + ".json", item.v_embedding_py, item.quality_py );
            embedder.createTemplate( test_data_path + "/images/" + item.img_file, item.v_embedding, item.quality );

            if( item.v_embedding_py.size() == 0 )
            {
                if( b_synchro ) item.face_id = -1; // skip c++
                mis_by_py++;
                item.face_id_py = -1;
                std::cout << "py skip file: " << item.img_file << std::endl;
            } // error calc embedding
            if( item.v_embedding.size() == 0 )
            {
                if( b_synchro ) item.face_id_py = -1; // skip c++
                mis_by_cc++;
                item.face_id = -1; std::cout << "c++ skip file: " << item.img_file << std::endl;
            } // error calc embedding

            if( (++n_counter) % percent_5 == 0 )
                std::cout << "processed:" << n_counter << std::endl;
        }
        std::cout << "processed files:" << n_counter  << " missed by python:" << mis_by_py << " missed by c++:" << mis_by_cc << std::endl;
        // calc and print statistics
        n_counter = 0;
        const int               log_step = 1000 * 1000;

        std::cout << "f_bound: " << f_bound << std::endl;

        for(size_t i = 0; i < test_dataset_size - 1; i++)
            for(size_t j = i + 1; j < test_dataset_size; j++)
            {
                n_matches_cc++;
                bool skip_match = false;

                int id_i = test_dataset[i].face_id;
                int id_j = test_dataset[j].face_id;

                if(id_i == 0 || id_j == 0)
                {
                    std::cout << "can not matching, found image without label" << std::endl;
                    return 0;
                }

                if(id_i < 0 )
                {
                    id_i *= -1;
                    skip_match = true;
                }

                if(id_j < 0)
                {
                    id_j *= -1;
                    skip_match = true;
                }

                double similarity = 0;
                if(!skip_match)
                {
                    if( embedder.matchTemplates( test_dataset[i].v_embedding, test_dataset[j].v_embedding, similarity, dist_type) < 0 )
                    {
                        std::cout << "matchTemplates failed" << std::endl;
                        return 0;
                    }

                    if(id_i == id_j)
                    {
                        matches_true_cc++;
                        if( dist_type == CFace2Embng::COS )
                        {
                            if( similarity < f_bound )
                                FP_cc++;
                            else
                                TP_cc++;
                        }
                        else
                        if( dist_type == CFace2Embng::EUCL )
                        {//
                            if( similarity > f_bound )
                                FP_cc++;
                            else
                                TP_cc++;
                        }
                    }
                    else
                    {
                        matches_false_cc++;
                        if( dist_type == CFace2Embng::COS )
                        {
                            if( similarity < f_bound )
                                TN_cc++;
                            else
                                FN_cc++;
                        }
                        else
                        if( dist_type == CFace2Embng::EUCL )
                        {//
                            if( similarity > f_bound )
                                TN_cc++;
                            else
                                FN_cc++;
                        }
                    }
                }
                else
                    skip_matches_cc++;

                n_counter++;

                if(n_counter % (10 * log_step) == 0)
                    std::cout << "match " << n_counter/log_step << "M descriptor pairs" << std::endl;
            }

        std::cout << "calc statistics for py" << std::endl;
        for(size_t i = 0; i < test_dataset_size - 1; i++)
            for(size_t j = i + 1; j < test_dataset_size; j++)
            {
                n_matches_py++;
                bool skip_match = false;

                int id_i = test_dataset[i].face_id_py;
                int id_j = test_dataset[j].face_id_py;

                if(id_i == 0 || id_j == 0)
                {
                    std::cout << "can not matching, found image without label" << std::endl;
                    return 0;
                }

                if(id_i < 0 )
                {
                    id_i *= -1;
                    skip_match = true;
                }

                if(id_j < 0)
                {
                    id_j *= -1;
                    skip_match = true;
                }

                double similarity = 0;
                if(!skip_match)
                {
                    if( embedder.matchTemplates( test_dataset[i].v_embedding_py, test_dataset[j].v_embedding_py, similarity, dist_type) < 0 )
                    {
                        std::cout << "matchTemplates failed" << std::endl;
                        return 0;
                    }

                    if(id_i == id_j)
                    {
                        matches_true_py++;
                        if( dist_type == CFace2Embng::COS )
                        {
                            if( similarity < f_bound )
                                FP_py++;
                            else
                                TP_py++;
                        }
                        else
                        if( dist_type == CFace2Embng::EUCL )
                        {//
                            if( similarity > f_bound )
                                FP_py++;
                            else
                                TP_py++;
                        }
                    }
                    else
                    {
                        matches_false_py++;
                        if( dist_type == CFace2Embng::COS )
                        {
                            if( similarity < f_bound )
                                TN_py++;
                            else
                                FN_py++;
                        }
                        else
                        if( dist_type == CFace2Embng::EUCL )
                        {//
                            if( similarity > f_bound )
                                TN_py++;
                            else
                                FN_py++;
                        }
                    }
                }
                else
                    skip_matches_py++;

                n_counter++;

                if(n_counter % (10 * log_step) == 0)
                    std::cout << "match " << n_counter/log_step << "M descriptor pairs" << std::endl;
            }
    }
    std::cout << "files count:" << n_files << std::endl;
    std::cout << "common statistics(incl c++):" << std::endl;
    std::cout << "matches: " << n_matches_cc << " skip_matches: " << skip_matches_cc << std::endl;
    std::cout << "matches_true: " << matches_true_cc << " matches_false: " << matches_false_cc << std::endl;
    std::cout << "TP: " << TP_cc << " FP: " << FP_cc << std::endl;
    std::cout << "TN: " << TN_cc << " FN: " << FN_cc << std::endl;

    double precision = 0.0;
    if(TP_cc > 0 || FP_cc > 0) precision = static_cast<double>(TP_cc) / (static_cast<double>(TP_cc) + static_cast<double>(FP_cc));

    double recall = 0.0;
    if( TP_cc > 0 || FN_cc > 0 ) recall = static_cast<double>(TP_cc) / (static_cast<double>(TP_cc) + static_cast<double>(FN_cc));

    double accuracy = 0.0;
    if(TP_cc > 0 || FP_cc > 0 || FN_cc > 0 || TN_cc > 0) accuracy = (static_cast<double>(TP_cc) + static_cast<double>(TN_cc)) / (static_cast<double>(TP_cc) + static_cast<double>(FP_cc) + static_cast<double>(TN_cc) + static_cast<double>(FN_cc));

    std::cout << "precision: " << precision << std::endl;
    std::cout << "recall: " << recall << std::endl;
    std::cout << "accuracy: " << accuracy << std::endl;

    std::cout << "matches: " << n_matches_py << " skip_matches: " << skip_matches_py << std::endl;
    std::cout << "matches_true: " << matches_true_py << " matches_false: " << matches_false_py << std::endl;
    std::cout << "TP: " << TP_py << " FP: " << FP_py << std::endl;
    std::cout << "TN: " << TN_py << " FN: " << FN_py << std::endl;

    std::cout << std::endl << "py statistics:" << std::endl;
    precision = 0.0;
    if(TP_py > 0 || FP_py > 0) precision = static_cast<double>(TP_py) / (static_cast<double>(TP_py) + static_cast<double>(FP_py));

    recall = 0.0;
    if( TP_py > 0 || FN_py > 0 ) recall = static_cast<double>(TP_py) / (static_cast<double>(TP_py) + static_cast<double>(FN_py));

    accuracy = 0.0;
    if(TP_py > 0 || FP_py > 0 || FN_py > 0 || TN_py > 0) accuracy = (static_cast<double>(TP_py) + static_cast<double>(TN_py)) / (static_cast<double>(TP_py) + static_cast<double>(FP_py) + static_cast<double>(TN_py) + static_cast<double>(FN_py));

    std::cout << "precision: " << precision << std::endl;
    std::cout << "recall: " << recall << std::endl;
    std::cout << "accuracy: " << accuracy << std::endl;

    return 0;
}
