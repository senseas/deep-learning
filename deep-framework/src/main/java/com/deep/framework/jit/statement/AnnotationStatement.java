package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

import static javax.lang.model.SourceVersion.isIdentifier;

@Data
public class AnnotationStatement implements Statement {
    public List<Object> packages = new ArrayList<>();

    public void parser(Statement parent, Object obj, List<Object> list) {
        Parser.statements.add(this);
        Object o = list.remove(0);
        while (true) {
            o = list.remove(0);
            if (o instanceof String && isIdentifier((String) o)) {
                packages.add(o);
                return;
            }
        }
    }

}
