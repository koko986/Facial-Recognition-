create table if not exists people (
  id text primary key,
  name text not null,
  created_at text not null
);

create table if not exists face_images (
  id text primary key,
  person_id text references people(id) on delete cascade,
  storage_path text not null,
  image_kind text not null check (image_kind in ('training', 'original', 'compressed')),
  svd_rank integer,
  created_at text not null
);

create table if not exists compression_experiments (
  id text primary key,
  person_id text references people(id) on delete set null,
  person_name text not null,
  svd_rank integer not null,
  original_size_bytes integer not null,
  compressed_size_bytes integer not null,
  compression_ratio real not null,
  storage_reduction_percent real not null,
  mse real not null,
  psnr real not null,
  processing_time_ms real not null,
  predicted_person_id text references people(id) on delete set null,
  predicted_name text not null,
  confidence real not null,
  accepted integer not null,
  recognition_method text not null,
  original_image_path text not null,
  compressed_image_path text not null,
  created_at text not null
);
