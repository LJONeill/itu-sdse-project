package main

import (
	"context"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	Build()
	data_pipeline()
	//ml_pipeline()

	if err != nil {
		return err
	}
	return nil
}

func Build(ctx context.Context) error {
	
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("src")).
		WithExec([]string{"python", "--version"})

	python = python.WithExec([]string{"python", "config.py"})

	_, err = python.
		Directory("build_output"). // Before writing to this folder, we may have to make sure it exists, 'os.makedirs('build_output', exist_ok=True)' 
		Export(ctx, "build_output") //the exist_ok=True makes nothing happen if it already exists
	if err != nil {
		return err
	}
	return nil
}

func data_pipeline() {
	
	data_pipeline := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("src")).
		WithExec([]string{"python", "dataset.py"})

	//data_pipeline = data_pipeline.WWithExec([]string{"python", "features.py"})

	_, err = data_pipeline.
		Directory("data_pipeline_output"). // See above re: folder creation
		Export(ctx, "data_pipeline_output")
	if err != nil {
		return err
	}
	return nil
}

func ml_pipeline() {
	
	ml_pipeline := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("src/modelling")).
		WithExec([]string{"python", "train.py"})
		
	ml_pipeline = ml_pipeline.WWithExec([]string{"python", "model_selection.py"})

	_, err = ml_pipeline.
		Directory("ml_pipeline_output"). // See above re: folder creation
		Export(ctx, "ml_pipeline_output")
	if err != nil {
		return err
	}
	return nil
}
