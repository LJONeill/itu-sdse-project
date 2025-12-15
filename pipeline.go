package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	if err := Build(ctx); err != nil {
        fmt.Println("Error:", err)
        panic(err)
    }
	//ml_pipeline()
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Mirror the root of our repository
	itu_sdse_project := client.Host().Directory(".")

	// Before running any py files, install requirements
	require := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("/repo", itu_sdse_project).
		WithWorkdir("/repo/src").
		WithExec([]string{"python", "--version"})

	require = require.WithExec([]string{
		"bash", "-lc",
		"pip install --upgrade pip",
	})
	
		require = require.WithExec([]string{
		"bash", "-lc",
		"python -m pip install -r /repo/requirements.txt",
	})
	_, err = require.Stdout(ctx)
	if err != nil {
		return err
	}

	python := require.WithExec([]string{"python", "config.py"})

	_, err = python.
		Directory("output"). 
		Export(ctx, "output")
	if err != nil {
		return err
	}

	data := require.WithExec([]string{"python", "dataset.py"})

	//data = data.WWithExec([]string{"python", "features.py"})

	_, err = data.
		Directory("output"). // See above re: folder creation
		Export(ctx, "output")
	if err != nil {
		return err
	}
	return nil
}