export type RecognitionResult = {
  predicted_person_id: string | null;
  predicted_name: string;
  confidence: number;
  accepted: boolean;
  method: string;
};

export type ExperimentResult = {
  id: string;
  person_id: string | null;
  person_name: string;
  svd_rank: number;
  original_size_bytes: number;
  compressed_size_bytes: number;
  compression_ratio: number;
  storage_reduction_percent: number;
  mse: number;
  psnr: number;
  processing_time_ms: number;
  recognition: RecognitionResult;
  original_image_url: string;
  compressed_image_url: string;
  created_at: string;
};

export type AnalyzeResponse = {
  original_recognition: RecognitionResult;
  results: ExperimentResult[];
  recommended_rank: number | null;
  accuracy_threshold: number;
};

export type Person = {
  id: string;
  name: string;
  created_at: string;
  image_count: number;
};
