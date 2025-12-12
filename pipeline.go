package main

import (
	"context"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	Build()
	data_pipeline()
	ml_pipeline()

	

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

	python = python.WithExec([]string{"python", "__init__.py"})

	_, err = python.
		Directory("output")
		Export(ctx, "output")
	if err != nil {
		return err
	}

	return nil
}

func data_pipeline() {
	
	data_pipeline := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("src")).
		WithExec([]string{"python", "dataset.py"})

	_, err = data_pipeline.
		Directory("output")
		Export(ctx, "output")
	if err != nil {
		return err
	}
}

func ml_pipeline() {
	
	ml_pipeline := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("itu-forked-project", client.Host().Directory("src/modelling")).
		WithExec([]string{"python", "train.py"})
		
	ml_pipeline = ml_pipeline.WWithExec([]string{"python", "model_selection.py"})

	_, err = ml_pipeline.
		Directory("output")
		Export(ctx, "output")
	if err != nil {
		return err
	}
}
